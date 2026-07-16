# -*- coding: utf-8 -*-
"""envs 包: safety-gymnasium 安全约束环境 (numpy 单 env + CPU 向量化(串行/多进程) + 论文 config0-3 复刻)。"""
from . import paper_envs                                     # import 即注册 4 个论文 env id
from .paper_envs import PAPER_ENV_IDS, make_paper_env
from .safety_env import SafetyEnv
from .safety_env_vec import SafetyVecEnv
from .safety_env_vec_mp import SafetyVecEnvMP, make_vec_env

__all__ = ['SafetyEnv', 'SafetyVecEnv', 'SafetyVecEnvMP', 'make_vec_env',
           'PAPER_ENV_IDS', 'make_paper_env']
