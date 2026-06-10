import torch
import torch.nn as nn
from einops import repeat


class TPosePrior(nn.Module):
    def __init__(self, file_path: str = "checkpoints/t_pose.pt"):
        super().__init__()
        t_pose = torch.load(file_path, weights_only=True, map_location="cpu").to(torch.float32)
        self.register_buffer("t_pose", t_pose, persistent=False)

    def forward(
        self,
        assignment_xyzs: torch.Tensor,  # (B, N, D_last) with vis flag in last dim
        *args,
    ):
        joint_num = self.t_pose.shape[0]
        offset = repeat(assignment_xyzs[..., :3], "b n d -> b n j d", j=joint_num)

        prior_ref_poses_xyz = self.t_pose.unsqueeze(0).unsqueeze(0) + offset

        ret_val = {
            "prior_poses_xyz": prior_ref_poses_xyz,
        }

        return ret_val
