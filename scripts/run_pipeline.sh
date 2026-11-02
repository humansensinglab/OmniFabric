#!/bin/bash
# OmniFabric inference: single garment image -> simulation-ready sewing pattern with a normalized texture.
#
# Usage:
#   bash scripts/run_pipeline.sh <input_image> [output_dir]
#
# Environment variables (all optional):
#   GPU             GPU index to use                                   (default: 0)
#   TEXTURE_BACKEND how the textured frontal view is generated: flux2 (FLUX.2-dev, automatic) or gemini (manual: the
#                   inputs and prompt are prepared in <output_dir>/<garment>/gemini/, you generate the image in the
#                   Gemini app and save it into gemini/result/)   (default: ask when run from a terminal, else flux2)
#   VIEWS_BACKEND   minimax360 (front/back/left/right from a 360 orbit video, needs a running ComfyUI server,
#                   see scripts/start_comfyui.sh) or flux2 (back view only, FLUX.2)   (default: minimax360)
#   COMFY_SERVER    ComfyUI server address                             (default: http://127.0.0.1:8188)
#   COMFYUI_DIR     ComfyUI root, for its input/ and output/ folders   (default: third_party/ComfyUI)
#   CHATGARMENT_CKPT   ChatGarment weights      (default: checkpoints/chatgarment/pytorch_model.bin)
#   OMNIFABRIC_CKPT    OmniFabric LoRA directory (default: checkpoints/omnifabric)
#   SEED            seed for the generative steps                      (default: 42)
#   GARMENTS        which predicted garments to texture, e.g. "lower" or "upper lower" (upper / lower / wholebody)
#                   or "all". If unset and several garments are predicted, the script asks when run from a terminal
#                   and textures all of them otherwise.
#
# Outputs (per predicted garment, e.g. valid_garment_upper):
#   <output_dir>/<garment>/coarse_texture.png        coarse texture map T' projected from the generated views
#   <output_dir>/<garment>/normalized_texture.png    OmniFabric normalized texture T (full resolution, dilated)
#   <output_dir>/<garment>/final/                    simulated 3D garment with the normalized texture
#
# Re-running with the same output_dir resumes: the sewing pattern prediction, the neutral simulation and the generated
# views are reused if they exist (delete them to regenerate); all later steps are recomputed.

set -e

INPUT_IMAGE=$(realpath "$1")
NAME=$(basename "${INPUT_IMAGE%.*}")
OUT=$(realpath -m "${2:-outputs/$NAME}")

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export GARMENTCODE_ROOT="${GARMENTCODE_ROOT:-$REPO_ROOT/third_party/GarmentCodeRC}"
export EGL_DEVICE_ID=${GPU:-0}
# keep the output to progress bars, questions and results: hide library warnings and info logs
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error DIFFUSERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false

GPU=${GPU:-0}
if [[ -z "$TEXTURE_BACKEND" ]]; then [[ -t 0 ]] && TEXTURE_BACKEND=ask || TEXTURE_BACKEND=flux2; fi
VIEWS_BACKEND=${VIEWS_BACKEND:-minimax360}
COMFY_SERVER=${COMFY_SERVER:-http://127.0.0.1:8188}
export CHATGARMENT_CKPT=$(realpath -m "${CHATGARMENT_CKPT:-checkpoints/chatgarment/pytorch_model.bin}")
OMNIFABRIC_CKPT=$(realpath -m "${OMNIFABRIC_CKPT:-checkpoints/omnifabric}")
SEED=${SEED:-42}

run() { CUDA_VISIBLE_DEVICES=$GPU python -m "$@"; }

python -m omnifabric.utils.check_checkpoints --output_dir "$OUT" \
    --texture_backend "$TEXTURE_BACKEND" --views_backend "$VIEWS_BACKEND" \
    --chatgarment_ckpt "$CHATGARMENT_CKPT" --omnifabric_ckpt "$OMNIFABRIC_CKPT" --comfy_server "$COMFY_SERVER"

mkdir -p "$OUT/input"
[[ "$INPUT_IMAGE" -ef "$OUT/input/$(basename "$INPUT_IMAGE")" ]] || cp "$INPUT_IMAGE" "$OUT/input/"


# ===============================
# 1. Predict sewing patterns with ChatGarment and simulate them (GarmentCode)
echo "=== [1/6] Sewing pattern prediction ==="
export CHATGARMENT_OUTPUT_DIR="$OUT/chatgarment"
if [[ ! -f "$CHATGARMENT_OUTPUT_DIR/vis_new/all_json_spec_files.json" ]]; then
    # ChatGarment reads its docs/ and GarmentCode assets/ relative to its own root
    (cd third_party/ChatGarment && CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/omnifabric/sewing/predict_pattern.py" \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --model_name_or_path liuhaotian/llava-v1.5-7b \
        --version v1 \
        --data_path ./ \
        --data_path_eval "$OUT/input" \
        --image_folder ./ \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "$OUT/chatgarment/llava_args" \
        --per_device_eval_batch_size 1 \
        --model_max_length 3072 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --report_to none)
    run omnifabric.sewing.simulate --all_paths_json "$CHATGARMENT_OUTPUT_DIR"
fi

# ChatGarment may predict several garments (e.g. upper + lower): pick the ones to texture
# (GARMENTS, or an interactive prompt when run from a terminal, or all of them)
GARMENT_SPECS=$(python -m omnifabric.sewing.select_garments "$CHATGARMENT_OUTPUT_DIR/vis_new/all_json_spec_files.json" \
    --garments "$GARMENTS")


for GARMENT_SPEC in $GARMENT_SPECS; do
    GARMENT_SPEC=$(realpath "$GARMENT_SPEC")
    GARMENT_NAME=$(basename "$GARMENT_SPEC" | sed 's/_specification\.json$//')
    G="$OUT/$GARMENT_NAME"
    SIM="$G/sim/$GARMENT_NAME"          # body+garment renders, *_sim.obj and masks
    VIEWS="$G/views"
    mkdir -p "$G" "$VIEWS"
    echo ""
    echo "################ $GARMENT_NAME ################"


    # ===============================
    # 2. UV template, position map and a light-grey simulation with garment-only renders
    echo "=== [2/6] UV template, position map, neutral simulation ==="
    if [[ ! -f "$SIM/${GARMENT_NAME}_sim.obj" ]]; then
        python -m omnifabric.sewing.uv_template "$GARMENT_SPEC" --output_dir "$G" --template_name editable_template.png
        python -m omnifabric.sewing.position_map "$GARMENT_SPEC" --output_dir "$G"
        run omnifabric.sewing.textured_sim "$GARMENT_SPEC" assets/lightgrey.png --output_dir "$G/sim_tmp"
        rm -rf "$G/sim" && mv "$G/sim_tmp/${GARMENT_NAME}_textured" "$G/sim" && rmdir "$G/sim_tmp"
    fi


    # ===============================
    # 3. Pose-aware texture transfer: re-texture the frontal garment render from the input image
    echo "=== [3/6] Texture transfer onto the frontal render ==="
    python -m omnifabric.utils.crop_ref --input "$SIM/${GARMENT_NAME}_render_front.png" \
        --output "$SIM/${GARMENT_NAME}_render_front_cropped.png" --ratio 0.85 --vertical_offset -140
    python -m omnifabric.utils.color_background "$SIM/${GARMENT_NAME}_render_front_cropped.png" \
        "$SIM/${GARMENT_NAME}_render_front_cropped_chroma_key.png"
    python -m omnifabric.utils.color_background "$G/sim/${GARMENT_NAME}_render_garment_only_front.png" \
        "$VIEWS/render_front_chroma_key.png"

    if [[ ! -f "$VIEWS/front_garment.png" ]]; then
        # FLUX.2 (automatic) or Gemini (manual hand-off in $G/gemini/; exits with code 3 while waiting for the image)
        python -m omnifabric.coarse.texture_transfer_gemini --mode "$TEXTURE_BACKEND" --garment_name "$GARMENT_NAME" \
            --grey "$VIEWS/render_front_chroma_key.png" --reference "$INPUT_IMAGE" \
            --handoff_dir "$G/gemini" --output "$VIEWS/front_garment.png"
        [[ -f "$VIEWS/front_garment.png" ]] || run omnifabric.coarse.texture_transfer \
            --task texture-transfer \
            --image1 "$VIEWS/render_front_chroma_key.png" \
            --image2 "$INPUT_IMAGE" \
            -o "$VIEWS/front_garment.png" \
            --seed "$SEED" \
            --guidance 5.0 \
            --steps 25
    fi

    # textured frontal rendering R_f (garment composited onto the mannequin render)
    python -m omnifabric.utils.composite_garment_texture \
        --mannequin "$SIM/${GARMENT_NAME}_render_front_cropped_chroma_key.png" \
        --mask "$SIM/${GARMENT_NAME}_render_front_cropped_mask.png" \
        --reference "$VIEWS/front_garment.png" \
        --output "$VIEWS/front_human.png"
    python -m omnifabric.utils.remove_background "$SIM/${GARMENT_NAME}_render_front_cropped.png" \
        "$VIEWS/front_human.png" "$G/frontal_rendering.png"


    # ===============================
    # 4. Multi-view generation and non-overlapping projection onto the simulated mesh
    echo "=== [4/6] Multi-view generation ($VIEWS_BACKEND) and projection ==="
    PROJ_IN="$G/projection/inputs/$GARMENT_NAME"
    mkdir -p "$PROJ_IN"
    render_only() { echo "$G/sim/${GARMENT_NAME}_render_garment_only_$1.png"; }

    python -m omnifabric.utils.fit_view "$(render_only front)" "$VIEWS/front_garment.png" "$PROJ_IN/front_view.png" --aligned

    if [[ "$VIEWS_BACKEND" == "minimax360" ]]; then
        ls "$VIEWS"/orbit/*_back.png > /dev/null 2>&1 || python -m omnifabric.coarse.orbit_views "$VIEWS/front_garment.png" \
            --out_dir "$VIEWS/orbit" --server "$COMFY_SERVER" --seed "$SEED"
        # Orbit views are named after the garment's own left/right; renders after the viewer's left/right.
        for pair in back:back left:right right:left; do
            ORBIT_VIEW=$(ls "$VIEWS"/orbit/*_"${pair%%:*}".png | head -n 1)
            python -m omnifabric.utils.fit_view "$(render_only "${pair##*:}")" "$ORBIT_VIEW" "$PROJ_IN/${pair##*:}_view.png"
        done
    elif [[ "$VIEWS_BACKEND" == "flux2" ]]; then
        [[ -f "$VIEWS/back_garment.png" ]] || run omnifabric.coarse.back_view_flux2 --image "$VIEWS/front_garment.png" \
            -o "$VIEWS/back_garment.png" --width 1072 --height 1072 --seed 469
        python -m omnifabric.utils.remove_background "$(render_only back)" "$VIEWS/back_garment.png" "$PROJ_IN/back_view.png"
    else
        echo "Unknown VIEWS_BACKEND: $VIEWS_BACKEND" && exit 1
    fi

    cp "$SIM/${GARMENT_NAME}_sim.obj" "$PROJ_IN/mesh.obj"
    run omnifabric.coarse.project_multiview --input_root "$G/projection/inputs" \
        --output_root "$G/projection/outputs" --example_name "$GARMENT_NAME"


    # ===============================
    # 5. Coarse texture map T' in sewing pattern UV space
    echo "=== [5/6] UV texture extraction ==="
    python -m omnifabric.coarse.extract_uv_texture \
        --obj_path "$PROJ_IN/mesh.obj" \
        --glb_path "$G/projection/outputs/results_multiview_general_${GARMENT_NAME}_full/mesh_multiview_full_${GARMENT_NAME}_textured_mesh.glb" \
        --template_path "$G/editable_template.png" \
        --output_path "$G/coarse_texture.png"


    # ===============================
    # 6. OmniFabric texture normalization, then simulate the normalized sewing pattern
    echo "=== [6/6] Texture normalization and final simulation ==="
    # elongated layouts are cut between panels so the coarse texture is not shrunk on the 1024 canvas: very elongated
    # ones (ratio > 3) are stacked into a squarer layout, others are split into two parts normalized separately
    # (normalize/inputs/part0, part1); the normalized texture is put back into the original layout afterwards
    python -m omnifabric.normalize.square_layout to_square --template "$G/editable_template.png" \
        --coarse "$G/coarse_texture.png" --pos "$G/${GARMENT_NAME}_position_map.pt" --out_dir "$G/normalize/inputs"
    NORMALIZED_PARTS=()
    for PART_DIR in "$G"/normalize/inputs/part*; do
        PART=$(basename "$PART_DIR")
        run omnifabric.normalize.infer \
            --checkpoint "$OMNIFABRIC_CKPT" \
            --mask_path "$PART_DIR/editable_template.png" \
            --proj_mask_path "$PART_DIR/coarse_texture.png" \
            --pos_path "$PART_DIR/position_map.pt" \
            --ref_path "$G/frontal_rendering.png" \
            --prompt "A textured sewing pattern" \
            --output "$G/normalize/$PART/normalized.png" \
            --use_proj_mask \
            --target_size 1024 \
            --ref_size 512 \
            --save_conditions \
            --postprocess \
            --use_sr \
            --seed "$SEED" \
            --denoise_full
        NORMALIZED_PARTS+=("$G/normalize/$PART/normalized_fullres.png")
    done

    python -m omnifabric.normalize.square_layout from_square "${NORMALIZED_PARTS[@]}" \
        --layout "$G/normalize/inputs/layout.json" --output "$G/normalize/normalized_fullres_restored.png"
    python -m omnifabric.normalize.dilate_uv_panels "$G/normalize/normalized_fullres_restored.png" "$G/editable_template.png" \
        --N 10 --heal_radius 1 --output "$G/normalized_texture.png"

    run omnifabric.sewing.textured_sim "$GARMENT_SPEC" "$G/normalized_texture.png" --output_dir "$G/final_tmp"
    rm -rf "$G/final" && mv "$G/final_tmp/${GARMENT_NAME}_textured" "$G/final" && rmdir "$G/final_tmp"

    echo "Done: $G/final"
done
