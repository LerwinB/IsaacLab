# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka Hand environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

inhand_task_entry = "isaaclab_tasks.direct.hand_grasp"

gym.register(
    id="Isaac-Repose-Cube-Franka-Direct-v0",
    entry_point=f"{inhand_task_entry}.hand_grasp_env:HandGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_hand_env_cfg:FrankaHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Franka1-Direct-v0",
    entry_point=f"{inhand_task_entry}.panda_grasp_env:PandaGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_panda_env_cfg:FrankaPandaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Franka-OpenAI-FF-Direct-v0",
    entry_point=f"{inhand_task_entry}.hand_grasp_env:HandGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_hand_env_cfg:FrankaHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaHandAsymFFPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ff_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Franka-OpenAI-LSTM-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Franka_hand_env_cfg:FrankaHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

### Vision

gym.register(
    id="Isaac-Repose-Cube-Franka-Vision-Direct-v0",
    entry_point=f"{__name__}.Franka_hand_vision_env:FrankaHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Franka_hand_vision_env:FrankaHandVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Franka-Vision-Direct-Play-v0",
    entry_point=f"{__name__}.Franka_hand_vision_env:FrankaHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Franka_hand_vision_env:FrankaHandVisionEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)
