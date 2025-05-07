# Zhao Shengyun.
# All rights reserved.


"""Configuration for the dexterous hand.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand with implicit actuator model.


"""


import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

HAND2ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Factory/hand2arm.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, -0.7071, 0.7071),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            stiffness=1e6,
            damping=1e3,
            friction=0.0,
            armature=0.0,
            effort_limit=87,
            velocity_limit=124.6,
        ),
        # "panda_arm2": ImplicitActuatorCfg(
        #     joint_names_expr=["panda_joint[6-7]"],
        #     stiffness=0.0,
        #     damping=0.0,
        #     friction=0.0,
        #     armature=0.0,
        #     effort_limit=12,
        #     velocity_limit=149.5,
        # ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["f(1|2|3|4|5)(4|3|2|1)_joint"],
            effort_limit={
                "f(1|2|3|4)3_joint": 0.7245,
                "f1(1|2)_joint": 0.9,
                "f2(1|2)_joint": 0.9,
                "f3(1|2)_joint": 0.9,
                "f4(4|1|2)_joint": 0.9,
                "f54_joint": 2.3722,
                "f53_joint": 1.45,
                "f5(2|1)_joint": 0.99,
            },
            stiffness={
                "f(1|2|3|4|5)(3|2|1|4)_joint": 1.0,
            },
            damping={
                "f(1|2|3|4|5)(3|2|1|4)_joint": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""