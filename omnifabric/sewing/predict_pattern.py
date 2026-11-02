"""
Predict sewing patterns from garment images with ChatGarment (LLaVA-1.5 + LoRA), and convert them into GarmentCode
specifications (<CHATGARMENT_OUTPUT_DIR>/vis_new/valid_garment_<image>/, listed in vis_new/all_json_spec_files.json).
Run from third_party/ChatGarment with the arguments in scripts/run_pipeline.sh.
"""
import json
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import transformers
from easydict import EasyDict as edict
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict

logging.disable(logging.INFO)  # DeepSpeed, imported by ChatGarment, logs its setup at INFO level
from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.garment_utils_v2 import run_garmentcode_parser_float50
from llava.json_fixer import repair_json
from llava.mm_utils import tokenizer_image_token
from llava.model import GarmentGPTFloat50ForCausalLM
from llava.train.train_garmentcode_outfit import ModelArguments, DataArguments, TrainingArguments
logging.disable(logging.NOTSET)

from omnifabric.paths import REPO_ROOT
from omnifabric.utils.quiet import quiet

os.environ["MASTER_PORT"] = "23499"


def find_all_linear_names(model, lora_target_modules=['q_proj', 'v_proj']):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                [
                    x not in name
                    for x in [
                        'mm_projector', 'vision_tower', 'vision_resampler', 'float_layer'
                    ]
                ]
            )
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))



class LazyImageDataset(Dataset):
    """The .png/.jpg images of a folder, preprocessed for the CLIP vision tower."""

    def __init__(self, imagefolder: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 max_len=-1):
        super(LazyImageDataset, self).__init__()
        self.imagefolder = imagefolder
        self.all_images = [item for item in os.listdir(imagefolder) if item.endswith(('.png', '.jpg'))]
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        image_file = os.path.join(self.imagefolder, self.all_images[i])
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        return {'image': image, 'image_path': os.path.join(image_folder, image_file)}


def translate_args(model_args, data_args, training_args):
    args = edict(
        local_rank=local_rank,
        precision="bf16",
        model_max_length=training_args.model_max_length,
        vision_tower=model_args.vision_tower,
        log_base_dir='./runs',
        exp_name="try_lr1e_4_generator_wildimg",
        gradient_checkpointing=training_args.gradient_checkpointing,
        use_mm_start_end=False
    )
    return args


def main(args):
    try:
        import flash_attn  # noqa: F401
        attn_implementation = 'flash_attention_2'
    except ImportError:
        attn_implementation = 'sdpa'
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    args = translate_args(model_args, data_args, training_args)

    assert training_args.bits not in [4, 8]
    assert model_args.vision_tower is not None
    assert 'mpt' not in model_args.model_name_or_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    model = GarmentGPTFloat50ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        seg_token_idx=args.seg_token_idx,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False
    assert not model_args.freeze_backbone

    assert training_args.gradient_checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    assert model_args.version == "v1"
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates: # conv_vicuna_v1
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    assert not model_args.tune_mm_mlp_adapter
    assert not training_args.freeze_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    assert args.precision == "bf16"
    model = model.bfloat16().cuda()

    val_dataset = LazyImageDataset(
        tokenizer=tokenizer,
        imagefolder=data_args.data_path_eval,
        data_args=data_args,
    )

    # ChatGarment weights and output folder (set by scripts/run_pipeline.sh)
    resume_path = os.environ.get('CHATGARMENT_CKPT', os.path.join(REPO_ROOT, 'checkpoints/chatgarment/pytorch_model.bin'))
    state_dict = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.bfloat16().cuda()
    device = model.device

    if data_args.data_path_eval[-1] == '/':
        data_args.data_path_eval = data_args.data_path_eval[:-1]
    if data_args.data_path_eval.split('/')[-1] == 'img' or data_args.data_path_eval.split('/')[-1] == 'imgs':
        dataset_name = data_args.data_path_eval.split('/')[-2]
    else:
        dataset_name = data_args.data_path_eval.split('/')[-1]

    args.exp_name = resume_path.split('/')[-2]
    parent_folder = os.environ.get('CHATGARMENT_OUTPUT_DIR') or os.path.join(args.log_base_dir, args.exp_name, f'{dataset_name}_img_recon')
    os.makedirs(parent_folder, exist_ok=True)

    # Inference mode: the model was built for training (get_peft_model leaves the LoRA dropout active), so without
    # eval() every run predicts a different sewing pattern for the same image. Decoding is greedy; the seeds cover
    # anything else that draws random numbers.
    model.eval()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    all_json_spec_files = []
    # optional extra constraints for the prediction, e.g. "no hoodies"
    guidance_txt = os.environ.get("GUIDANCE_TXT", "").strip()

    for i in range(len(val_dataset)):
        data_item = val_dataset[i]

        # 1. describe the garments (JSON), 2. predict the sewing pattern from the image and that description
        question1 = 'Can you describe the geometry features of the apparel worn by the model in the Json format?  \
            For full-body garments like dresses, specify both upper body and skirt/bottom components. And please do not inclue waistband.'
        question2 = 'Can you estimate the sewing pattern code based on the image and description? For dresses, ensure meta.upper and meta.bottom are both specified.\
            And please do not inclue waistband.'
        questions = [question1, question2]

        for k in range(len(questions)):
            conv = conversation_lib.conv_templates[model_args.version].copy()
            conv.messages = []
            if k == 0:
                prompt = question1
                if guidance_txt:
                    prompt += f"\nPlease follow these constraints: {guidance_txt}"
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            else:
                extra = f"\nPlease follow these constraints: {guidance_txt}" if guidance_txt else ""
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + question2 + extra + "\n" + text_output.replace('upper_garment', 'upperbody_garment').replace('lower_garment', 'lowerbody_garment')

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_clip = data_item['image']
            image_clip = image_clip.unsqueeze(0).to(device)
            assert args.precision == "bf16"
            image_clip = image_clip.bfloat16()
            image = image_clip

            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).to(device)

            output_ids, float_preds, seg_token_mask = model.evaluate(
                image_clip,
                image,
                input_ids,
                max_new_tokens=2048,
                tokenizer=tokenizer,
            )

            output_ids = output_ids[0, 1:]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False).strip().replace("</s>", "")

            if k == 0:
                continue

            text_output = text_output.replace('[STARTS]', '').replace('[SEG]', '').replace('[ENDS]', '')
            image_path = data_item['image_path']
            garment_id = image_path.split('/')[-1].split('.')[0]
            json_output = repair_json(text_output, return_objects=True)

            output_dir = os.path.join(parent_folder, 'vis_new', f'valid_garment_{garment_id}')
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'output.txt'), 'w') as f:
                f.write(prompt + '\n' + text_output + '\n' + str(json_output))
            shutil.copy(image_path, os.path.join(output_dir, 'gt_image.png'))

            with quiet():
                all_json_spec_files = run_garmentcode_parser_float50(all_json_spec_files, json_output, float_preds,
                                                                     output_dir)

    with open(os.path.join(parent_folder, 'vis_new', 'all_json_spec_files.json'), 'w') as f:
        json.dump(all_json_spec_files, f)


if __name__ == "__main__":
    main(sys.argv[1:])

