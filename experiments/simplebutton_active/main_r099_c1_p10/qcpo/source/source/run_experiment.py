"""Frozen SimpleButton constructor settings for the two final recurrent agents."""
import json
from pathlib import Path
from types import SimpleNamespace
from agents.four_env_fullbatch_formal import DQC_LR2_KL_ALGO, QREF_FULL_ALGO

def base_args(algorithm, seed, device, environment):
    if environment != 'SimpleButton':
        raise ValueError('This artifact contains only SimpleButton.')
    method = {DQC_LR2_KL_ALGO: 'dca', QREF_FULL_ALGO: 'qcpo'}[algorithm]
    settings = json.loads((Path(__file__).resolve().parents[1] / 'configs' / (method + '_constructor.json')).read_text())
    settings.update(seed=int(seed), device=device)
    if 'experiment_variant' in settings:
        settings['experiment_variant'] = settings['experiment_variant'].replace('-s0-', f'-s{seed}-')
    return SimpleNamespace(**settings)
