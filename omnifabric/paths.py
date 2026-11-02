"""Locations of repo resources and third-party code (override with environment variables)."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(REPO_ROOT, 'third_party')

GARMENTCODE_ROOT = os.path.abspath(os.environ.get('GARMENTCODE_ROOT', os.path.join(THIRD_PARTY, 'GarmentCodeRC')))
SIM_PROPS = os.path.join(REPO_ROOT, 'configs', 'sim_props.yaml')
