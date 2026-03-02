import torch
import h5py
import torch.nn as nn
from sklearn.decomposition import PCA
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import gzip, struct
from sklearn.cluster import KMeans
import sys
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.eval.utils import ModelWithIntermediateLayers
import torch.distributed as dist
#from dinov3.eval.linear import LinearClassifier
import json


num_classes = 2
device = 'cuda:3'
linear_ckp = '../../../PANDA-checkpoints/checkpoint_pretrain_vitL16_1792/eval/training_499999/Linear_ParticleClassification/checkpoints/best/checkpoint.pth'


if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
        init_method="tcp://127.0.0.1:29500"
    )

classification_path = "../../../PANDA-checkpoints/checkpoint_pretrain_vitL16_1792/eval/training_499999/Linear_ParticleClassification/"
# Point to your ViT-L/16 config and checkpoint
model_cfg = ModelConfig(
    config_file=f"{classification_path}config.yaml",
    pretrained_weights=f"{('/').join(classification_path.split('/')[:-2])}/teacher_checkpoint.pth"
)

model, ctx = load_model_and_context( model_cfg, output_dir="./")
# model is ready for inference; ctx['autocast_dtype'] gives the suggested precision

n_last_blocks = 1

ckpt = torch.load(f"{('/').join(classification_path.split('/')[:-2])}/teacher_checkpoint.pth", map_location="cpu")

state = {}
for k, v in ckpt["teacher"].items():
    if k.startswith("backbone."):
        state[k[len("backbone."):]] = v
    else:
        state[k] = v

missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing:", missing)
print("Unexpected:", unexpected)

autocast_ctx = partial(torch.autocast, device_type=device, enabled=True)
feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
feature_model.to(device).eval()

head = nn.Linear(2 * model.embed_dim, num_classes)
head_ckpt = torch.load(linear_ckp, map_location='cpu')
ck = head_ckpt["linear_classifiers"]

w = ck["module.classifiers_dict.classifier_1_blocks_avgpool_True_lr_0_02500.linear.weight"]
b = ck["module.classifiers_dict.classifier_1_blocks_avgpool_True_lr_0_02500.linear.bias"]

with torch.no_grad():
    head.weight.copy_(w)
    head.bias.copy_(b)

head.to(device).eval()

window = 256
NORMALIZE_MEAN = 0.5475857690770103
NORMALIZE_STD = 0.12047245612891117

image = MRC('/storage/cryoppp/10028/micrographs/062.mrc').data[:,:,0]

# image_ = (image/255-NORMALIZE_MEAN)/NORMALIZE_STD
image_ = torch.from_numpy(np.copy(image)).unsqueeze(0).unsqueeze(0).float()
#resized_image =  F.interpolate(image_, size=(np.array(image.shape)*(224/window)).astype(int), mode="bicubic")[0]#mode='bilinear')[0]#mode="nearest")[0]
#resized_image = (resized_image-resized_image.min())/(resized_image.max()-resized_image.min())
# resized_image = (resized_image-NORMALIZE_MEAN)/NORMALIZE_STD
# Min-max normalize to 0–1 as float32
image_norm = ((image - image.min()) / (image.max() - image.min()))
image_norm = image_norm.astype(np.float32)

# Convert to uint8 0–255
image_uint8 = (image_norm * 255).astype(np.uint8)

# Save as grayscale
#Image.fromarray(image_uint8, mode='L').save("micrograph.png")

#image_ = torch.from_numpy(np.copy((image_norm-NORMALIZE_MEAN)/NORMALIZE_STD)).unsqueeze(0).unsqueeze(0).float()
dv = 64
im_s = 224
im_sh = 128
resize_shape = tuple((np.round(np.array(image.shape)*1 / dv)*dv).astype(int))
resized_image =  F.interpolate(image_, size=resize_shape, mode="bicubic")[0][0]

H, W = resized_image.shape
rss_x = (H) // dv
rss_y = (H) // dv
if dv<im_s:
    rss_x = (H-im_s) // dv
    rss_y = (H-im_s) // dv

# # (H,W) → (rss_x, 224, rss_y, 224)
# patches_np = (
#     resized_image.numpy()
#     .reshape(rss_x, 224, rss_y, 224)
#     .transpose(0, 2, 1, 3)          # (rss_x, rss_y, 224, 224)
#     .reshape(-1, 224, 224)          # (N, 224, 224)
# )
# im_sm = torch.from_numpy(patches_np).float()
# im_sm = torch.from_numpy(create_submaps(resized_image,224,dv))
# im_sm.shape

im_sm = []
for ix in range(rss_x):
    #print(dv*ix, dv*ix+im_s)
    for iy in range(rss_y):
        if dv*ix+im_s>H or dv*iy+im_s>W:
            print(ix, iy)
            continue
        w = resized_image[dv*ix:dv*ix+im_s,dv*iy:dv*iy+im_s]
        # if w.shape != (im_s, im_s):
        #     w = resized_image[dv*ix-im_s:dv*ix,dv*iy-im_s:dv*iy]
        im_sm.append(w)
im_sm = torch.from_numpy(np.array(im_sm)).float()
im_sm.shape

batch_size = 1024
latents = []
patches = im_sm.float()   # shape: (N, 224, 224)

# import h5py
# with h5py.File('/mnt/extdisk/azamanos_data/particles_h5/zbackground_particles.h5') as f:
#     patches = torch.from_numpy(f['images'][:1000]/255).float()

patches = patches.unsqueeze(1)              # shape: (N, 1, 224, 224)

patches_min = patches.amin(dim=(-1,-2), keepdims=True)
patches_max = patches.amax(dim=(-1,-2), keepdims=True)
patches = (patches-patches_min)/(patches_max-patches_min)
patches = 1-patches
patches = (patches-NORMALIZE_MEAN)/NORMALIZE_STD

N = patches.shape[0]

with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch = patches[start:end].to(device)     # shape: (B  , 1, 224, 224)
            # forward pass
            latents.append(create_linear_input(feature_model(batch), use_n_blocks=1, use_avgpool=True).detach().cpu())
            print(f'{start}/{N}', end='\r')
latents = torch.cat(latents, dim=0)  # shape: (N, latent_dim)

classify = []
with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch = latents[start:end].to(device)     # shape: (B  , 1, 224, 224)

            # forward pass
            classify.append(head(batch).detach().cpu())
classify = torch.cat(classify, dim=0)#.numpy()   # shape: (N, latent_dim)

im = classify.softmax(1)[:,1]
im = (im-im.min())/(im.max()-im.min())

#image = im_sm.numpy()#.reshape(resize_shape)
counter = 0
proj_tensor = torch.zeros((rss_x, rss_y))
for ix in range(rss_x):
    for iy in range(rss_y):
        proj_tensor[ix:(ix+1),iy:(iy+1)] = im[counter]
        counter+=1
proj_tensor = proj_tensor.unsqueeze(0).unsqueeze(0).float()
###
tp, lr = ((H // dv)-rss_x)//2, ((W // dv)-rss_y)//2
if tp:
    proj_tensor = F.pad(proj_tensor, (lr, lr, tp, tp), mode='constant', value=0)
proj_up = F.interpolate(proj_tensor, size=image.shape, mode="nearest")[0][0].numpy()#mode='bilinear')[0]#mode="nearest")[0]
###
Image.fromarray((proj_up)*255).convert('L').save(f'prediction_*1_{dv}.png')

heatmap = classify.softmax(1).argmax(1)
heatmap = heatmap.reshape(rss_x,rss_y)
proj_up = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0).float(), size=image.shape, mode="nearest")[0][0].numpy()
Image.fromarray((proj_up)*255).convert('L').save(f'prediction_*1_{dv}_hard.png')

print(classify.softmax(1).argmax(1).sum())
