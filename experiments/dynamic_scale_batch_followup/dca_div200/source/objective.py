"""Only the reward discount varies. All cost targets and multiplier updates use raw episode costs."""
import os
import torch

REWARD_GAMMA = 1.0
if REWARD_GAMMA not in (1.0, 0.99):
    raise ValueError("SIMPLEBUTTON_REWARD_GAMMA must be 1.0 or 0.99")
COST_GAMMA = 1.0
COST = "undiscounted_environment_cost"
EVENT = COST + "_strict_gt_15"

def powers(values, gamma):
    return gamma ** torch.arange(len(values), device=values.device, dtype=torch.float64)[:, None]

def episode_total(values, gamma):
    return (values.double() * powers(values, gamma)).sum(0)
