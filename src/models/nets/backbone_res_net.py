import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision import models

from src.utils.camera import pixel_to_camera
from src.utils.linear_algebra import affine_transform, transform_to_original_img_space
from src.utils.method import top_k_heatmaps

log = logging.getLogger(__name__)


class PoseResNet(nn.Module):
    """Human pose estimation model based on ResNet backbone.

    Reuses torchvision's ResNet implementation and adds deconvolution
    layers for upsampling to generate spatial heatmaps for joint detection.
    """

    def __init__(
        self,
        njoints=15,
        max_num_people=10,
        root_idx=2,
        conf_threshold=0.2,
        num_deconv_filters=(256, 256, 256),
        num_deconv_kernels=(4, 4, 4),
        final_conv_kernel=1,
        deconv_with_bias=False,
        backbone="resnet50",
        ckpt_path: str | None = None,
        root_heatmap_noise_scale: float = 1.0,
    ):
        super().__init__()

        self.deconv_with_bias = deconv_with_bias
        self.max_num_people = max_num_people
        self.njoints = njoints
        self.root_joint_idx = root_idx
        self.score_threshold = conf_threshold
        self.root_heatmap_noise_scale = root_heatmap_noise_scale

        # Build the torchvision ResNet backbone. With a checkpoint we overwrite
        # these weights below, so init randomly; otherwise prefer ImageNet weights
        # but fall back to random init if they can't be fetched (offline / no
        # cached download) so construction never hard-fails.
        resnets = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        }
        if backbone not in resnets:
            raise ValueError(f"Unsupported backbone: {backbone}")
        ctor, imagenet_weights = resnets[backbone]
        if ckpt_path is not None:
            self.backbone = ctor(weights=None)
        else:
            try:
                self.backbone = ctor(weights=imagenet_weights)
            except Exception as exc:  # offline / SSL / no cached download
                log.warning("Could not fetch ImageNet weights for %s (%s); using random init.", backbone, exc)
                self.backbone = ctor(weights=None)

        # Remove the global average pooling and fc layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Get the number of output channels from the backbone
        if backbone in ["resnet18", "resnet34"]:
            self.inplanes = 512
        else:
            self.inplanes = 2048

        # Build deconvolution layers
        self.deconv_layers = self._make_deconv_layer(
            len(num_deconv_filters),
            num_deconv_filters,
            num_deconv_kernels,
        )

        # Final layer to produce heatmaps
        self.final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=njoints,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

        # Initialize deconv and final layers
        self._init_deconv_layers()

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.load_ckpt(ckpt_path)

    def _get_deconv_cfg(self, deconv_kernel):
        """Calculate padding and output padding for deconvolution."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Unsupported deconv kernel size: {deconv_kernel}")

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Create deconvolution layers."""
        assert num_layers == len(num_filters), "Number of layers must match number of filters"
        assert num_layers == len(num_kernels), "Number of layers must match number of kernels"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels=self.inplanes,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=self.deconv_with_bias,
                    ),
                    nn.BatchNorm2d(planes, momentum=0.1),
                    nn.ReLU(inplace=True),
                ]
            )
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _init_deconv_layers(self):
        """Initialize weights for deconvolution and final layers."""
        # Initialize deconv layers
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize final layer
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)

    def detect_root_joints(self, root_heatmap):
        """
        Detect root joint locations from heatmap.

        Args:
            root_heatmap: (B*A*T*V, 1, H, W) - root joint heatmap

        Returns:
            instances_hm: (B*A*T*V, N, 1, 3) - detected root joints in heatmap space (x, y, score)
        """
        ind_k, val_k = top_k_heatmaps(root_heatmap, K=self.max_num_people)
        ind_k = rearrange(ind_k, "n b ... -> b n ...").float()
        val_k = rearrange(val_k, "n b ... -> b n ...").float()

        # Apply threshold mask
        valid_mask = val_k > self.score_threshold  # (B_flat, N)
        ind_k = torch.where(valid_mask, ind_k, torch.zeros_like(ind_k))
        val_k = torch.where(valid_mask, val_k, torch.zeros_like(val_k))

        instances_hm = torch.cat([ind_k, val_k], dim=-1)

        return instances_hm

    def project_to_original_image(self, instances_hm, centers, scales, rotations, heatmap_size):
        """
        Project heatmap coordinates to original image space.

        Args:
            instances_hm: (B_flat, N, 1, 3) - instances in heatmap space
            centers: (B_flat, 2)
            scales: (B_flat, 2)
            rotations: (B_flat,)
            heatmap_size: (W, H)

        Returns:
            instances_img: (B_flat, N, 1, 3) - instances in original image space
        """
        B_flat, N, J, _ = instances_hm.shape

        # Reshape for transform_preds_torch
        instances_flat = rearrange(instances_hm, "b n j d -> b (n j) d")  # (B_flat, N*J, 3)

        # Transform coordinates
        instances_img = transform_to_original_img_space(
            instances_flat,
            centers,
            scales,
            rotations,
            heatmap_size,
        )

        # Reshape back
        instances_img = rearrange(instances_img, "b (n j) d -> b n j d", n=N, j=J)

        return instances_img

    def forward(self, batch, interm_feat_levels=(0, 1, 2)):
        """Create feature maps from input images."""
        # (B, A, T, V, C, H, W)
        sources = batch["source"]
        extra_dimensions = sources.shape[:-3]
        sources_flat = sources.flatten(0, -4)  # (B*A*T*V, C, H, W)

        x = self.backbone(sources_flat)
        intermediate_features = []
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d):
                intermediate_features.append(x)
        heatmaps = self.final_layer(x)

        all_features = [f for (i, f) in enumerate(intermediate_features) if i in interm_feat_levels] + [heatmaps]
        all_features = all_features[::-1]
        all_features = [feat.view(*extra_dimensions, *feat.shape[1:]) for feat in all_features]

        return {
            "feature_maps": all_features,
            "heatmaps": heatmaps,
            "extra_dimensions": extra_dimensions,
        }

    def create_instances(self, batch, heatmaps, mode="train"):
        """Detect and process instances from heatmaps."""
        B, A, T, V, C, H, W = batch["source"].shape
        device = batch["source"].device

        H_hm, W_hm = heatmaps.shape[2:4]
        N = self.max_num_people
        J = self.njoints

        # Extract root joint heatmap
        root_heatmap = heatmaps[:, self.root_joint_idx : self.root_joint_idx + 1, :, :]  # (B*A*T*V, 1, H, W)

        # Detect root joints in heatmap space
        instances_hm_root = self.detect_root_joints(root_heatmap)  # (B*A*T*V, N, 1, 3)

        # Project to original image space
        centers = batch["center"].flatten(0, -2)  # (B*A*T*V, 2)
        scales = batch["scale"].flatten(0, -2)  # (B*A*T*V, 2)
        rotations = batch["rotation"].flatten()  # (B*A*T*V,)

        # Reshape to (B, A, T, V, N, 1, 3) for further processing
        instances_hm_root = instances_hm_root.view(B, A, T, V, N, 1, 3)

        # ---- Get person IDs by matching predictions to ground truth ----
        person_ids = self.align_pred_to_gt(batch, instances_hm_root)  # (B, A, T, V, N, 1)

        if mode == "train":
            if self.root_heatmap_noise_scale > 0:
                noise = torch.rand_like(instances_hm_root[..., :2]) * (W_hm / 30) * self.root_heatmap_noise_scale
                instances_hm_root[..., :2] += noise
            
            # Set score to 0 where person_id == -1
            # We don't want to train on invalid matches
            invalid_mask = person_ids == -1  # (B, A, T, V, N, 1)
            instances_hm_root[..., 2][invalid_mask] = 0.0

        # --- Project predictions to original image space ---
        instances_img_root = self.project_to_original_image(
            rearrange(instances_hm_root, "b a t v n j d -> (b a t v) n j d"),
            centers,
            scales,
            rotations,
            (W_hm, H_hm),
        )
        instances_img_root = instances_img_root.view(B, A, T, V, N, 1, 3)

        # --- Compute normalized camera coordinates ---
        # (B, A, T, V, N, 1, 2)
        instances_norm_cc_root = pixel_to_camera(
            instances_img_root[..., :2],
            repeat(batch["cam_params_vec"], "... d -> ... n j d", n=N, j=1),
        )

        # (B, A, T, V, N)
        instances_scores_root = instances_img_root[..., 0, -1].clone()

        # Initialize all tensors with zeros
        instances_hm_full = torch.zeros(B, A, T, V, N, J, 3, device=device, dtype=torch.float32)
        instances_img_full = torch.zeros(B, A, T, V, N, J, 3, device=device, dtype=torch.float32)
        instances_scores_full = torch.zeros(B, A, T, V, N, J, 1, device=device, dtype=torch.float32)
        instances_norm_cc_full = torch.zeros(B, A, T, V, N, J, 2, device=device, dtype=torch.float32)

        # Place root joint data at index 2
        instances_hm_full[:, :, :, :, :, self.root_joint_idx, :] = instances_hm_root.squeeze(-2)
        instances_img_full[:, :, :, :, :, self.root_joint_idx, :] = instances_img_root.squeeze(-2)
        instances_scores_full[:, :, :, :, :, self.root_joint_idx, 0] = instances_scores_root
        instances_norm_cc_full[:, :, :, :, :, self.root_joint_idx, :] = instances_norm_cc_root.squeeze(-2)

        instance_dict = {
            "heat_map": instances_hm_full,  # (B, A, T, V, N, J, 3)
            "image": instances_img_full,  # (B, A, T, V, N, J, 3)
            "score": instances_scores_full,  # (B, A, T, V, N, J, 1)
            "norm_cc": instances_norm_cc_full,  # (B, A, T, V, N, J, 2)
            "person_ids": person_ids,  # (B, A, T, V, N, 1)
        }

        return instance_dict

    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def align_pred_to_gt(self, batch, instances_hm_root):
        """
        Vectorized matching of predicted root joints to GT root joints.

        Only aligns predictions to GT slots that are:
          - visible at the root joint
          - AND have person_id != -1 (i.e. matched)

        Args:
            batch: Dict containing data
            instances_hm_root: (B, A, T, V, N, 1, 3) in heatmap space

        Returns:
            person_ids_pred: (B, A, T, V, N, 1) - GT person index for each prediction slot,
                             or -1 if no matched GT.
        """
        gt_kps_xys = batch["gt_keypoints_xys"]  # (B, A, T, V, N, J, 3)
        affine = batch["affine_transforms"]  # (B, A, T, V, 2, 3)

        B, A, T, V, N, J, _ = gt_kps_xys.shape
        K = B * A * T * V  # combined (b,a,t,v) dimension

        # --- GT person IDs ---
        gt_pids = batch["person_ids"]  # (B, A, T, V, N)

        # --- Extract GT root joints in original image space (2D annotations) ---
        gt_root_xy_orig = gt_kps_xys[..., self.root_joint_idx, :2]  # (B, A, T, V, N, 2)
        gt_root_score = gt_kps_xys[..., self.root_joint_idx, 2]  # (B, A, T, V, N)

        # flatten for affine transform: (K, N, 2)
        gt_root_xy_orig_flat = rearrange(gt_root_xy_orig, "b a t v n d -> (b a t v) n d")
        affine_flat = rearrange(affine, "b a t v x y -> (b a t v) x y")  # (K, 2, 3)

        # transform to input space and then to heatmap space
        gt_root_xy_input_flat = affine_transform(gt_root_xy_orig_flat, affine_flat)  # (K, N, 2)
        gt_root_xy_hm_flat = gt_root_xy_input_flat / 4.0  # (K, N, 2)

        # combine with visibility score -> (K, N, 3)
        gt_root_score_flat = rearrange(gt_root_score, "b a t v n -> (b a t v) n 1")
        gt_roots = torch.cat([gt_root_xy_hm_flat, gt_root_score_flat], dim=-1)  # (K, N, 3)

        # visible roots
        gt_valid = gt_roots[..., 2] > 0  # (K, N) bool

        # --- restrict to *matched* GT only (pid != -1) ---
        gt_pids_flat = rearrange(gt_pids, "b a t v n -> (b a t v) n")  # (K, N)
        matched_mask = gt_pids_flat != -1
        gt_valid = gt_valid & matched_mask  # (K, N) bool

        # --- Pred roots in heatmap space ---
        # instances_hm_root: (B, A, T, V, N, 1, 3)
        pred_roots = rearrange(instances_hm_root, "b a t v n 1 d -> (b a t v) n d")  # (K, N, 3)
        pred_valid = pred_roots[..., 2] > self.score_threshold  # (K, N)

        # 4. Pairwise distances (K, N_pred, N_gt)
        dists = torch.cdist(pred_roots[..., :2], gt_roots[..., :2])  # (K, N, N)

        # 5. Apply validity & distance threshold
        valid_pair_mask = pred_valid.unsqueeze(2) & gt_valid.unsqueeze(1)  # (K, N, N)

        dist_threshold = 8.0
        invalid_mask = (~valid_pair_mask) | (dists > dist_threshold)
        dists.masked_fill_(invalid_mask, float("inf"))

        # 6. Greedy one-to-one matching per K
        pred_to_gt_ids = torch.full((K, N), -1, dtype=torch.long, device=dists.device)
        work_dists = dists.clone()

        for _ in range(N):
            flat = work_dists.view(K, -1)  # (K, N*N)
            min_vals, flat_inds = flat.min(dim=1)  # per K

            is_valid_match = min_vals < float("inf")
            if not is_valid_match.any():
                break

            valid_k = torch.nonzero(is_valid_match).squeeze(1)  # indices of K with valid match

            # decompose linear index into (pred_idx, gt_idx)
            p_idx = flat_inds[valid_k] // N
            g_idx = flat_inds[valid_k] % N

            # assign match
            pred_to_gt_ids[valid_k, p_idx] = g_idx

            # invalidate used rows/cols for these K
            work_dists[valid_k, p_idx, :] = float("inf")
            work_dists[valid_k, :, g_idx] = float("inf")

        # 7. Map GT indices back to person_ids (so we return GT person IDs, not slot indices)
        matched_gt_pids = torch.full_like(pred_to_gt_ids, -1)
        # only for pred positions that matched some gt index
        has_match = pred_to_gt_ids != -1
        matched_gt_pids[has_match] = gt_pids_flat.gather(1, pred_to_gt_ids.clamp(min=0).long())[has_match].long()

        # reshape to (B, A, T, V, N)
        person_ids_pred = rearrange(matched_gt_pids, "(b a t v) n -> b a t v n 1", b=B, a=A, t=T, v=V)

        return person_ids_pred

    def load_ckpt(self, path_to_model: str):
        """Load a checkpoint from a given path."""
        ckpt = torch.load(path_to_model, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        try:
            filtered_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith("backbone")}
            ret_val = self.load_state_dict(filtered_state_dict, strict=True)
        except RuntimeError:
            ret_val = self.remap(state_dict)
        return ret_val

    def remap(self, in_sd):
        """
        This function handles the specific naming conventions observed between a standard
        PyTorch ResNet backbone and the custom structure of the PoseFlow backbone.
        """
        poseflow_state_dict = OrderedDict()

        for key, value in in_sd.items():
            new_key = key

            # --- Key Transformation Logic ---

            # 1. Handle the 'keypoint_head' prefix removal
            if new_key.startswith("keypoint_head."):
                new_key = new_key.replace("keypoint_head.", "")

            # 2. Handle the main backbone layers
            # The mapping is: layer1 -> 4, layer2 -> 5, layer3 -> 6, layer4 -> 7
            elif new_key.startswith("backbone.layer"):
                if "layer1" in new_key:
                    new_key = new_key.replace("layer1", "4")
                elif "layer2" in new_key:
                    new_key = new_key.replace("layer2", "5")
                elif "layer3" in new_key:
                    new_key = new_key.replace("layer3", "6")
                elif "layer4" in new_key:
                    new_key = new_key.replace("layer4", "7")

            # 3. Handle the initial conv1 and bn1 layers
            elif new_key.startswith("backbone.conv1"):
                new_key = new_key.replace("conv1", "0")
            elif new_key.startswith("backbone.bn1"):
                new_key = new_key.replace("bn1", "1")

            poseflow_state_dict[new_key] = value

        return self.load_state_dict(poseflow_state_dict, strict=True)
