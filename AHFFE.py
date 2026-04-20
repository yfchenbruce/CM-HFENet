import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AHFFE(nn.Module):
    def __init__(self,
                 ratio: tuple[float, float] = (0.4, 0.4),
                 patch: tuple[int, int] = (8, 8),
                 use_filtering: bool = True,   # Ablation switch: Adaptive High-Pass Filter (AHPF)
                 use_spatial: bool = True,     # Ablation switch: Spatial Feature Perception Path
                 use_channel: bool = True,     # Ablation switch: Channel Feature Perception Path
                 groups: int = 32              # Default value, can be omitted in YAML
                 ) -> None:
        """
        Adaptive High-Frequency Feature Enhancement Module (AHFFE)
        
        :param ratio: Initial cutoff ratio (rh, rw) for the AHF filter
        :param patch: Pooling size for spatial/channel paths (ph, pw)
        :param use_filtering: Whether to enable Adaptive High-Pass Filter (AHPF)
        :param use_spatial: Whether to enable the spatial enhancement path
        :param use_channel: Whether to enable the channel enhancement path
        """
        super().__init__()
        self.ph, self.pw = int(patch[0]), int(patch[1])
        self.use_filtering = use_filtering
        self.use_spatial = use_spatial
        self.use_channel = use_channel
        self.groups = int(groups)
        
        # --- Adaptive Parameter Initialization ---
        # Clamp input ratio to prevent numerical instability in log
        safe_r0 = min(max(ratio[0], 0.01), 0.99)
        safe_r1 = min(max(ratio[1], 0.01), 0.99)
        
        # Initialize according to the paper: θ = log(r / (1 - r))
        init_val_h = math.log(safe_r0 / (1 - safe_r0))
        init_val_w = math.log(safe_r1 / (1 - safe_r1))
        
        # Learnable parameters [H_param, W_param]
        self.ratio_param = nn.Parameter(torch.tensor([init_val_h, init_val_w], dtype=torch.float32))
        
        # Component placeholders for lazy initialization
        self._built = False
        self._c = None
        self.spatial_conv: nn.Module | None = None
        self.channel_conv1: nn.Module | None = None
        self.channel_conv2: nn.Module | None = None
        self.out_conv: nn.Module | None = None

    def _build_if_needed(self, x: torch.Tensor) -> None:
        """Lazy initialization: build layers based on input channel dimension and ablation switches"""
        c = x.shape[1]
        if self._built and self._c == c:
            return
        
        device = x.device
        g = max(1, min(self.groups, c))
        
        # Spatial enhancement path
        if self.use_spatial:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            ).to(device)
        
        # Channel enhancement path
        if self.use_channel:
            self.channel_conv1 = nn.Conv2d(c, c, kernel_size=1, groups=g).to(device)
            self.channel_conv2 = nn.Conv2d(c, c, kernel_size=1, groups=g).to(device)
        
        # Final output convolution + normalization
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, c), num_channels=c),
        ).to(device)
        
        self._built = True
        self._c = c

    def _mask_fft(self, x: torch.Tensor) -> torch.Tensor:
        """Core of Adaptive High-Frequency Filter (AHF Filter)"""
        B, C, H, W = x.shape
        W_rfft = W // 2 + 1
        
        # 1. Forward rFFT
        xf = torch.fft.rfft2(x, dim=(-2, -1))
        
        # 2. Get current learned cutoff ratios (range ≈ (0, 1))
        current_ratios = torch.sigmoid(self.ratio_param)
        rh, rw = current_ratios[0], current_ratios[1]

        # 3. Generate normalized coordinate grids [0, 1]
        grid_h = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
        grid_w = torch.linspace(0, 1, W_rfft, device=x.device).view(1, 1, 1, W_rfft)

        # 4. Differentiable soft mask (temperature=20)
        mask_h = torch.sigmoid((grid_h - rh) * 20)
        mask_w = torch.sigmoid((grid_w - rw) * 20)
        
        # 5. Combined mask: suppress low-frequency region (top-left)
        mask = 1.0 - ((1.0 - mask_h) * (1.0 - mask_w))

        # 6. Apply mask and inverse rFFT
        xf = xf * mask
        xh = torch.fft.irfft2(xf, s=(H, W))
        return xh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build_if_needed(x)
            
        # Step 1: Frequency-domain filtering (AHPF)
        if self.use_filtering:
            feat_to_process = self._mask_fft(x)
        else:
            feat_to_process = x
        
        # Step 2: Spatial enhancement path (SEP)
        if self.use_spatial:
            spa = self.spatial_conv(feat_to_process) * x
        else:
            spa = torch.zeros_like(x)
        
        # Step 3: Channel enhancement path (CEP)
        if self.use_channel:
            # Pooling + ReLU + spatial summation
            amax = F.adaptive_max_pool2d(feat_to_process, output_size=(self.ph, self.pw))
            aavg = F.adaptive_avg_pool2d(feat_to_process, output_size=(self.ph, self.pw))
            
            amax = torch.sum(F.relu(amax), dim=(2, 3), keepdim=True)
            aavg = torch.sum(F.relu(aavg), dim=(2, 3), keepdim=True)
            
            # Group convolution for channel attention
            ch = self.channel_conv1(amax) + self.channel_conv1(aavg)
            ch = torch.sigmoid(self.channel_conv2(ch))
            cha = ch * x
        else:
            cha = torch.zeros_like(x)
        
        # Step 4: Fusion and output
        if not self.use_spatial and not self.use_channel:
            # Both paths disabled → directly transform the filtered feature
            out = self.out_conv(feat_to_process)
        else:
            # Normal case: fuse spatial and channel enhanced features
            out = self.out_conv(spa + cha)
            
        return out