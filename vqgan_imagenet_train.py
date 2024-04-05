import os
import yaml

from PIL import Image

import torch
import numpy as np

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader, Dataset


from taming.models.vqgan import VQModel

os.environ["TORCH_HOME"] = "/data/jdsu/ckpts/torch"
torch.cuda.set_device(1)

# load ckpt
ckpt_path = "./logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt"
# load model config
config_path = "./logs/vqgan_imagenet_f16_1024/configs/vqgan_imagenet_f16_1024.yaml"
config = OmegaConf.load(config_path)
# print(config["model"]["params"])
print(f"pretrained model config: \n{yaml.dump(OmegaConf.to_container(config.model.params))}")

# load vqgan model, include encoder, decoder, quantize, and loss module
# vqgan_model = VQModel(**config.model.params).eval()  # 900M
vqgan_model = VQModel(**config.model.params) # 900M
print(vqgan_model)


# train: load data
# this dataset only contains 1 image
class TestTrainSet(Dataset):
    def __init__(self, image_path, resolution=(256,256), length=2000):
        super().__init__()
        image = Image.open(image_path)
        image = image.resize(resolution)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.tensor(image).to(dtype=torch.float32)   # h,w,c
        # resize
        self.image = image  # naruto.png
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # assert index == 0, f"index out of range: {index}(index)/{1}(data_num)"
        return {"image": self.image}

def custom_collate(batch):
    image_tensor = [d["image"] for d in batch]
    image_tensor = torch.stack(image_tensor, 0, out=None)
    return {"image": image_tensor}

train_set = TestTrainSet("naruto.png")
train_loader = DataLoader(train_set, batch_size=12, shuffle=True, collate_fn=custom_collate)

# load pretrained weights and ignore decoder and discriminator
vqgan_model.init_from_ckpt(ckpt_path, ignore_keys=["decoder", "loss.discriminator"])
vqgan_model.freeze_pretrained_weights()

# train
trainer = Trainer(gpus=[1], max_epochs=50)
trainer.fit(vqgan_model, train_loader)
# save ckpt
trainer.save_checkpoint("checkpoints/reverse_30_epochs.ckpt")

