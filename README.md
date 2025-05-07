## IsaacLab
[documentation page](https://isaac-sim.github.io/IsaacLab)

[documentation page cn](https://docs.robotsfan.com/isaaclab/index.html)

## Git
http://172.17.175.66:3000/ShengyunZhao/IsaacLab.git

跨设备需要修改的路径 
1. `source\isaaclab\isaaclab\utils\assets.py`，NUCLEUS_ASSET_ROOT_DIR
2. `source\isaaclab_assets\isaaclab_assets\robots\frankahand.py`, ASSET_DIR
上传文件
```shell
scp -r Assets\Robots\FrankanoHand  buaa22@172.17.135.244:/home/buaa22/Software/zhaoshengyun/IsaacLab-main/Assets/Isaac/IsaacLab/Robots/FrankaNoHand
```
## scripts

```shell
# import urdf
python scripts/tools/convert_urdf.py Assets\nohand2arm_description\urdf\hand2arm.urdf Assets\Robots\FrankanoHand\FrankanoHand.usd --fix-base
```
```shell 
# tele operate
python scripts\environments\teleoperation\teleop_se3_agent.py --task Isaac-Reach-Franka-IK-Rel-v0
```
```shell
# train 
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Reach-Frankahand-v0 --headless
CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Repose-Cube-Franka-Direct-v0 --headless

python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Reach-Frankahand-v0 --headless --video --video_length 200 --checkpoint logs/rl_games/reach_franka/2025-03-21_15-17-20/nn/last_reach_franka_ep_1000_rew__-451756.9_.pth
```

## task1 reach
### 内容
1. 换机械臂
2. RL训不出来
3. 遥操作手部一直在随机乱动，**直接影响后续操作**
    1. 灵巧手手指有问题，除了中指，其他手指存在就会导致机械臂乱动

## task2 grasp
### 内容
1. 改造shadowhand inhand_manipulation 转方块代码
    1. code dir 
    ```shell
    source\isaaclab_tasks\isaaclab_tasks\direct\franka_hand
    source\isaaclab_tasks\isaaclab_tasks\direct\hand_grasp
    ```
2. 只保留中指和手掌，训练其触碰方块
    1. reward: 修改成功判定为方块移动一定的距离，
    2. obs=80, action=11
    3. 未知原因，场景总是重置(绝对坐标要转换为场景坐标)

