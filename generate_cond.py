import sys
import yaml

from PIL import Image
import numpy as np

import torch

from omegaconf import OmegaConf

from taming.models.cond_transformer import Net2NetTransformer

def save_image(s, name):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    s = Image.fromarray(s)
    s.save(f"./generate_imgs/{name}.png")
    print(f"save generate_imgs/{name}.png.")

# set path
sys.path.append(".")

# load config
config_path = "./logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml"
config = OmegaConf.load(config_path)
print(f"config yaml: \n{yaml.dump(OmegaConf.to_container(config))}")

# load model
model = Net2NetTransformer(**config.model.params)
ckpt_path = "./logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt"
state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("load model done.")
print(f"missing: {missing}")
print(f"unexpected: {unexpected}")
# cuda
torch.cuda.set_device(1)    # cuda:1
model.cuda().eval()     # 6326MiB, 6G

# load segmentation image
seg_name = "seg2"
segmentation_path = f"{seg_name}.png" # segmentation image
segmentation = Image.open(segmentation_path)
segmentation = np.array(segmentation)
segmentation = np.eye(182)[segmentation]
segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)
print(f"segmentation shape: {segmentation.shape}")
# resolution (683, 1024), segmentation channels: 182

# encode
seg_code, seg_indices = model.encode_to_c(segmentation) # encode to condition
print(f"seg_code: {seg_code.shape}")
print(f"seg_indices: {seg_indices.shape}")
assert seg_code.shape[0] * seg_code.shape[2] * seg_code.shape[3] == seg_indices.shape[0]
# segmentation_rec = model.cond_stage_model.decode(seg_code)  # reconstruct

# random indices
codebook_size = config.model.params.first_stage_config.params.n_embed
img_indices_shape = seg_indices.shape
img_code_shape = seg_code.shape # (batch, embed_dim, h, w)
# generate random indices
img_indices = torch.randint(codebook_size, img_indices_shape, device=model.device)
# random indices => random image
img_sample = model.decode_to_img(img_indices, img_code_shape)
# save image
save_image(img_sample, "random")

# generate image, from random to cond
img_indices = img_indices.reshape(img_code_shape[0], img_code_shape[2], img_code_shape[3])
seg_indices = seg_indices.reshape(seg_code.shape[0], seg_code.shape[2], seg_code.shape[3])
# generate image via auto regressive
temperature = 1.0
top_k = 100
update_every = 500
# generate high resolution image: sliding attention
for i in range(0, img_code_shape[2]):
    # loop for h, local i is the pos i in the patch
    if i <= 8:
        local_i = i
    elif img_code_shape[2] - i < 8:
        local_i = 16 - (img_code_shape[2] - i)
    else:
        local_i = 8
    for j in range(0, img_code_shape[3]):
        # loop for w
        if j <= 8:
            local_j = j
        elif img_code_shape[3] - j < 8:
            local_j = 16 - (img_code_shape[3] - j)
        else:
            local_j = 8
        # crop the patch
        i_start = i - local_i
        i_end = i_start + 16
        j_start = j - local_j
        j_end = j_start + 16
        img_patch = img_indices[:, i_start:i_end, j_start:j_end]
        img_patch = img_patch.reshape(img_patch.shape[0], -1)   # 2d => 1d (h, w) => (hw,)
        seg_patch = seg_indices[:, i_start:i_end, j_start:j_end]
        seg_patch = seg_patch.reshape(seg_patch.shape[0], -1)
        # concat the patch for predict (prepend the seg patch)
        patch = torch.cat([seg_patch, img_patch], dim=1)
        # predict the next token (get logits)
        logits, _ = model.transformer(patch[:, :-1])    # (batch, len, vocab_size)
        logits = logits[:, -256:, :]    # last 256, the first is the last seg token
        logits = logits.reshape(img_code_shape[0], 16, 16, -1)  # 1d => 2d
        logits = logits[:, local_i, local_j, :]     # predict index for (i, j), (batch, vocab_size)
        logits = logits / temperature
        if top_k is not None:
            logits = model.top_k_logits(logits, top_k)
        
        # logits to probs
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # set index to img_indices
        img_indices[:, i, j] = torch.multinomial(probs, num_samples=1)

        # record
        step = i * img_code_shape[3] + j
        if step % update_every == 0 or step == img_code_shape[2] * img_code_shape[3] - 1:
            # sample
            img_sample = model.decode_to_img(img_indices, img_code_shape)
            # save image
            save_image(img_sample, f"{seg_name}_step_{step}")
