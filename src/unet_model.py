import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        # Pad if dimension mismatch
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load resnet34 backbone from timm
        encoder = timm.create_model("resnet34", pretrained=pretrained)
        
        # Collapse 3-channel weights (64, 3, 7, 7) into 1-channel grayscale (64, 1, 7, 7) by averaging
        w_rgb = encoder.conv1.weight.data # [64, 3, 7, 7]
        w_gray = w_rgb.mean(dim=1, keepdim=True) # [64, 1, 7, 7]
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data.copy_(w_gray)
        self.bn1 = encoder.bn1
        self.act1 = encoder.act1
        self.maxpool = encoder.maxpool
        
        # Stages
        self.layer1 = encoder.layer1 # 64 ch, /4
        self.layer2 = encoder.layer2 # 128 ch, /8
        self.layer3 = encoder.layer3 # 256 ch, /16
        self.layer4 = encoder.layer4 # 512 ch, /32
        
        # Decoder
        # 512 (layer4) + 256 (layer3) -> 256
        self.up4 = UpBlock(512, 256, 256)
        # 256 + 128 (layer2) -> 128
        self.up3 = UpBlock(256, 128, 128)
        # 128 + 64 (layer1) -> 64
        self.up2 = UpBlock(128, 64, 64)
        # 64 + 64 (conv1) -> 32
        self.up1 = UpBlock(64, 64, 32)
        
        # Heads
        # Main filament mask head (logits)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # True Spine head (logits)
        self.spine_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Deep supervision heads at decoder levels 2 (up3 -> 128ch) and 3 (up2 -> 64ch)
        self.aux1 = nn.Conv2d(128, 1, kernel_size=1)
        self.aux2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Input: [B, 1, H, W]
        h, w = x.shape[2], x.shape[3]
        
        # Encoder
        x0 = self.act1(self.bn1(self.conv1(x))) # [B, 64, H/2, W/2]
        x1 = self.layer1(self.maxpool(x0))      # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)                    # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)                    # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)                    # [B, 512, H/32, W/32]
        
        # Decoder
        d4 = self.up4(x4, x3)                   # [B, 256, H/16, W/16]
        d3 = self.up3(d4, x2)                   # [B, 128, H/8, W/8]
        d2 = self.up2(d3, x1)                   # [B, 64, H/4, W/4]
        d1 = self.up1(d2, x0)                   # [B, 32, H/2, W/2]
        
        # Upsample d1 to full resolution [H, W]
        d0 = F.interpolate(d1, size=(h, w), mode="bilinear", align_corners=True)
        
        main_logits = self.final_conv(d0)
        spine_logits = self.spine_conv(d0)
        
        if self.training:
            aux1_logits = F.interpolate(self.aux1(d3), size=(h, w), mode="bilinear", align_corners=True)
            aux2_logits = F.interpolate(self.aux2(d2), size=(h, w), mode="bilinear", align_corners=True)
            return {
                "main": main_logits,
                "spine": spine_logits,
                "aux1": aux1_logits,
                "aux2": aux2_logits,
            }
        else:
            return {
                "main": main_logits,
                "spine": spine_logits
            }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet34(pretrained=True).to(device)
    dummy = torch.randn(2, 1, 512, 512).to(device)
    out = model(dummy)
    print("Model initialized successfully!")
    print(f"Device: {device}")
    for k, v in out.items():
        print(f"Output '{k}': shape {v.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M")
