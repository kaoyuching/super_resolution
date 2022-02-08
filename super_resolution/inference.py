import os
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageOps
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from .model import RRDBNet
from .dataset import LRDataset


# load model    
# Genmodel_path = "./models/RRDB_ESRGAN_x4_png_l.pth"
# Genmodel = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
# Genmodel.load_state_dict(torch.load(Genmodel_path, map_location="cpu")["model_state_dict"])


# load test data
# test_df = pd.read_json("./test_data.json", orient="records")
# test_df.head()


# create dataloader
# test_transform = A.Compose([ToTensor()])

# test_img_list = list(test_df['image'])
# test_dataset = LRDataset(root="/home/doriskao/project/super_resolution/test_data", img_list=test_img_list, transform=test_transform)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def sr_model(model_path):
    # model_path = "./models/RRDB_ESRGAN_x4_png_l.pth"
    Genmodel = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    Genmodel.load_state_dict(torch.load(model_path, map_location="cpu")["model_state_dict"])
    Genmodel.cpu()
    Genmodel.eval()
    return Genmodel

# inference and save upscale images
# save test images(x4)
def generate_sr_image(img_path, model):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = np.array(img)

    transform = A.Compose([ToTensor()])
    lr_aug = transform(image=img)
    lr_img = lr_aug['image']
    lr_img = torch.unsqueeze(lr_img, 0)

    with torch.no_grad():
        sr_img = model(lr_img)

    sr_img = sr_img.squeeze().permute(1, 2, 0).numpy()
    sr_img = cv2.GaussianBlur(sr_img, ksize=(0, 0), sigmaX=1.414/2, borderType=cv2.BORDER_REFLECT)
    sr_img = np.clip(sr_img, 0, 1)*255
    return sr_img

# i = 0
# Genmodel.cpu()
# Genmodel.eval()
# for lrimgs, imgname in test_dataloader:
    # with torch.no_grad():
        # outimg = Genmodel(lrimgs)

    # lrimgs = lrimgs.squeeze().permute(1, 2, 0).numpy()
    # outimg = outimg.squeeze().permute(1, 2, 0).numpy()
    # outimg = cv2.GaussianBlur(outimg, ksize=(0, 0), sigmaX=1.414/2, borderType=cv2.BORDER_REFLECT)
    # outimg = np.clip(outimg, 0, 1)*255
    # out_imgname = os.path.splitext(imgname[0])[0]
    # cv2.imwrite(f"./test_data_scale4/{out_imgname}x4.png", cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR))
