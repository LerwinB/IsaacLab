import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .ptv3 import PointTransformerV3
import numpy as np
from scipy.spatial.transform import Rotation as R


def build_ptv3_input(points: torch.Tensor, grid_size=0.01) -> dict:
    """
    将点云 [N, 3] 转为 ptv3 的输入字典。
    """
    N = points.shape[0]
    coords = points.clone()               # shape [N, 3]
    feat = points.clone()                 # 也可以是其他 feature
    batch = torch.zeros(N, dtype=torch.long,device=points.device)  # 全部属于 batch 0

    return {
        "coord": coords,      # [N, 3]
        "feat": feat,         # [N, 3]
        "batch": batch,       # [N]
        "grid_size": grid_size
    }

def random_wrist_pose(translation_range=0.1, rotation_range=np.pi):
    """
    生成随机的手腕位姿 SE(3) 4×4 矩阵。
    - translation_range: 平移的最大绝对值（单位：米）
    - rotation_range: 每个轴旋转角度的范围（单位：弧度）
    """
    # 随机平移向量（uniform）
    t = np.random.uniform(-translation_range, translation_range, size=(3,))
    
    # 随机欧拉角旋转（xyz 顺序）
    angles = np.random.uniform(-rotation_range, rotation_range, size=(3,))
    R_mat = R.from_euler('xyz', angles).as_matrix()

    # 构造 SE(3) 矩阵
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

class PTv3WithHead(nn.Module):
    def __init__(self, backbone, mlp_dims=(512 + 6, 256, 28)):
        super().__init__()
        self.backbone = backbone
        layers = []
        for i in range(len(mlp_dims)-1):
            layers += [
                nn.Linear(mlp_dims[i], mlp_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
        self.head = nn.Sequential(*layers)

    def forward(self, data_dict, T):
        point = self.backbone(data_dict)
        feat = point.feat              # shape: [B, D] → cls_mode=True
        feat = feat.mean(dim=0, keepdim=True)
        feat = feat.repeat(T.shape[0], 1)
        x = torch.cat([feat, T], dim=1)
        logits = self.head(x)
        return logits

class PTv3HandNet:
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}
        self.cfg = cfg
        self.backbone = PointTransformerV3(in_channels=3, cls_mode=True)
        self.mlp_dims = (512 + 6, 256, 28)
        self.model = PTv3WithHead(backbone=self.backbone, mlp_dims=self.mlp_dims)
        self.model.load_state_dict(torch.load(self.cfg.get("model_path", "ptv3_hand_model.pth"), map_location='cpu'))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.grid_size = 0.01

    def _preprocess_images(self, rgb_img: torch.Tensor, depth_img: torch.Tensor) -> torch.Tensor:
        """
        预处理输入图像。
        rgb_img: [B, H, W, 3] RGB 图像
        depth_img: [B, H, W, 1] 深度图像
        返回: 点云
        """
        B, H, W, _ = rgb_img.shape
        device = rgb_img.device
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        u = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
        v = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

        z = depth_img.squeeze(-1)  # [B, H, W]
        mask = z > 0  # [B, H, W]

        x = ((u - cx) * z / fx)[mask]  # [N_total]
        y = ((v - cy) * z / fy)[mask]
        z = z[mask]

        points_3d = torch.stack((x, y, z), dim=1)  # [N_total, 3]

        # 分批提取每个图像的点
        pointclouds = []
        for i in range(B):
            mask_i = mask[i]
            z_i = depth_img[i, ..., 0][mask_i]  # [Ni]
            u_i = u[i][mask_i]
            v_i = v[i][mask_i]
            x_i = (u_i - cx) * z_i / fx
            y_i = (v_i - cy) * z_i / fy
            points_i = torch.stack([x_i, y_i, z_i], dim=1)  # [Ni, 3]
            pointclouds.append(points_i)

        return pointclouds  # List[Tensor: (Ni, 3)]


    def step(self, rgb_img: torch.Tensor, depth_img: torch.Tensor, wrist_pose: torch.Tensor):
        """
        points: [N, 3] 点云数据
        wrist_pose: [B, 4, 4] 手腕位姿矩阵
        """
        points = self._preprocess_images(rgb_img, depth_img)
        assert points.ndim == 2 and points.shape[1] == 3, "points should be of shape [N, 3]"
        assert wrist_pose.ndim == 3 and wrist_pose.shape[1:] == (4, 4), "wrist_pose should be of shape [B, 4, 4]"
        
        # 构建 ptv3 输入
        data_dict = build_ptv3_input(points, grid_size=self.grid_size)
        
        # 将数据移动到设备上
        data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
        wrist_pose = wrist_pose.to(self.device)

        # 前向传播
        logits = self.model(data_dict, wrist_pose)
        return logits

