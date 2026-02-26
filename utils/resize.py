import torch
import numpy as np
import torch.nn.functional as F

def resize_torch(img_np, size=(224, 224), dtype_rt=torch.float64):
    img = torch.as_tensor(np.copy(img_np), dtype=dtype_rt)
    img = img.unsqueeze(1)#.unsqueeze(0)  # (1, 1, H, W)
    resized = F.interpolate(img, size=size, mode='bilinear', align_corners=False, antialias=True)
    return resized.squeeze(0).squeeze(0).squeeze(1)