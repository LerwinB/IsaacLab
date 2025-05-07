# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import  ISAACLAB_NUCLEUS_DIR, LOCAL_ASSET_DIR

##
# Configuration
##
ASSET_DIR = LOCAL_ASSET_DIR

FRANKAHAND_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/Robots/FrankaHand/hand2arm.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.0001,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "f(1|2|3|4|5)(4|3|2|1)_joint": 0.01,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["f(1|2|3|4|5)(4|3|2|1)_joint"],
            effort_limit={"f(1|2|3|4|5)(4|3|2|1)_joint": 0.9},
            velocity_limit={"f(1|2|3|4|5)(4|3|2|1)_joint": 0.0},
            stiffness={"f(1|2|3|4|5)(4|3|2|1)_joint": 1e6},
            damping={"f(1|2|3|4|5)(4|3|2|1)_joint": 1e6},
            # effort_limit={
            #     "f(1|2|3|4)3_joint": 0.7245,
            #     "f1(1|2)_joint": 0.9,
            #     "f2(1|2)_joint": 0.9,
            #     "f3(1|2)_joint": 0.9,
            #     "f4(4|1|2)_joint": 0.9,
            #     "f54_joint": 2.3722,
            #     "f53_joint": 1.45,
            #     "f5(2|1)_joint": 0.99,
            # },
            # stiffness={
            #     "f(1|2|3|4|5)(3|2|1|4)_joint": 1.0,
            # },
            # damping={
            #     "f(1|2|3|4|5)(3|2|1|4)_joint": 0.1,
            # },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FRANKANOHAND_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/Robots/FrankanoHand/FrankanoHand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "f21_joint": 0.0,
            "f22_joint": 0.0,
            "f23_joint": 0.0,
            "f24_joint": 0.0,


        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["f**_joint"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# config for hand grasp
FRANKA_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/Robots/FrankaHand/hand2arm.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=0.0001,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "f(1|2|3|4|5)(4|3|2|1)_joint": 0.01,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["f(1|2|3|4|5)(4|3|2|1)_joint"],
            effort_limit={"f(1|2|3|4|5)(4|3|2|1)_joint": 0.9},
            velocity_limit={"f(1|2|3|4|5)(4|3|2|1)_joint": 0.9},
            stiffness={"f(1|2|3|4|5)(4|3|2|1)_joint": 100},
            damping={"f(1|2|3|4|5)(4|3|2|1)_joint": 100},
            # effort_limit={
            #     "f(1|2|3|4)3_joint": 0.7245,
            #     "f1(1|2)_joint": 0.9,
            #     "f2(1|2)_joint": 0.9,
            #     "f3(1|2)_joint": 0.9,
            #     "f4(4|1|2)_joint": 0.9,
            #     "f54_joint": 2.3722,
            #     "f53_joint": 1.45,
            #     "f5(2|1)_joint": 0.99,
            # },
            # stiffness={
            #     "f(1|2|3|4|5)(3|2|1|4)_joint": 1.0,
            # },
            # damping={
            #     "f(1|2|3|4|5)(3|2|1|4)_joint": 0.1,
            # },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKAHAND_PANDA_HIGH_PD_CFG = FRANKAHAND_PANDA_CFG.copy()
FRANKAHAND_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKAHAND_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKAHAND_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKAHAND_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKAHAND_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
FRANKANOHAND_PANDA_HIGH_PD_CFG = FRANKANOHAND_PANDA_CFG.copy()
FRANKANOHAND_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKANOHAND_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKANOHAND_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKANOHAND_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKANOHAND_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0

"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
