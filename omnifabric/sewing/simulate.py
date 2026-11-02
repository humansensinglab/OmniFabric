"""Simulate sewing patterns on the mean body with GarmentCode (NVIDIA Warp).

Usage:
    python -m omnifabric.sewing.simulate --all_paths_json <chatgarment_output_dir>
    python -m omnifabric.sewing.simulate --json_spec_file <garment_specification.json> [--easy_texture_path texture.png]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import warp as wp

from omnifabric.paths import GARMENTCODE_ROOT, SIM_PROPS
from omnifabric.utils.quiet import quiet

sys.path.insert(1, GARMENTCODE_ROOT)
wp.config.quiet = True  # no Warp banner (GarmentCode initializes Warp on import)


def run_simultion_warp(pattern_spec, sim_config, output_path, easy_texture_path):
    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    import pygarment.data_config as data_config
    from pygarment.meshgen.sim_config import PathCofig

    props = data_config.Properties(sim_config)
    props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
    props.set_section_stats('render', render_time={})

    spec_path = Path(pattern_spec)
    garment_name, _, _ = spec_path.stem.rpartition('_')  # assuming ending in '_specification'

    paths = PathCofig(
        in_element_path=spec_path.parent,
        out_path=output_path,
        in_name=garment_name,
        body_name='mean_all',
        smpl_body=False,
        add_timestamp=False,
        system_path=os.path.join(GARMENTCODE_ROOT, 'system.json'),
        easy_texture_path=easy_texture_path
    )

    with quiet(progress=True):
        garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
        garment_box_mesh.load()
        garment_box_mesh.serialize(paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])
        props.serialize(paths.element_sim_props)

        run_sim(
            garment_box_mesh.name,
            props,
            paths,
            save_v_norms=False,
            store_usd=False,  # True slows the simulation down to CPU rates
            optimize_storage=False,
            verbose=False
        )
        props.serialize(paths.element_sim_props)


parser = argparse.ArgumentParser()
parser.add_argument("--all_paths_json", type=str, default='', help="ChatGarment output folder (simulate all garments)")
parser.add_argument("--json_spec_file", type=str, default='', help="a single garment specification")
parser.add_argument("--easy_texture_path", type=str, default='', help="texture to apply to the garment")
args = parser.parse_args()

if args.all_paths_json:
    with open(os.path.join(args.all_paths_json, 'vis_new/all_json_spec_files.json')) as f:
        garment_json_paths = json.load(f)
else:
    garment_json_paths = [args.json_spec_file]

# GarmentCodeRC's system.json resolves bodies/sim configs relative to its own root
garment_json_paths = [os.path.abspath(p) for p in garment_json_paths]
easy_texture_path = os.path.abspath(args.easy_texture_path) if args.easy_texture_path else ''
os.chdir(GARMENTCODE_ROOT)

for json_spec_file in garment_json_paths:
    json_spec_file = json_spec_file.replace('validate_garment', 'valid_garment')
    print(f"Simulating {os.path.basename(json_spec_file).replace('_specification.json', '')}")
    run_simultion_warp(json_spec_file, SIM_PROPS, os.path.dirname(json_spec_file), easy_texture_path=easy_texture_path)
