# src/ara_ppo.py — ARA-PPO components
# Antonis Leveidiotis | University of Piraeus
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import List, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Re-export all classes defined in notebook 04.
# When importing from this module, the notebook must have been run first
# OR this file is used standalone — in that case paste class definitions here.

__all__ = [
    'RegimeGatingLayer',
    'ResidualBlock',
    'RegimeAwareExtractor',
    'AdaptiveClipCallback',
    'RegimeDecomposedValueNet',
    '_ARAMlpExtractorWrapper',
    'ARAPPOPolicy',
    'ARA_PPO_HPARAMS',
    'make_ara_ppo',
]