import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BACKBONE=os.environ.get('PREFEVAL_BACKBONE','qwen2.5-3b')
NAMES={'qwen2.5-3b':'Qwen2.5-3B-Instruct','qwen2.5-7b':'Qwen2.5-7B-Instruct'}
if BACKBONE not in NAMES: raise ValueError(BACKBONE)
BASE=ROOT.parent/'models'/NAMES[BACKBONE]
