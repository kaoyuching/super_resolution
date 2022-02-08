import os
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageOps
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import functools
# albumentation
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

# Generator network: generate the fake data
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

# Residual block
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

    
# Residual in Residual dense block
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
    
# # RRDB net (scale 2)
# class RRDBNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(RRDBNet, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv3 = nn.Conv2d(nf+nf, nf, 3, 1, 1, bias=True)
#         self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         fea = self.conv_first(x)
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         fea = fea + trunk

# #         fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea1 = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='bilinear')))
# #         fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea2 = self.lrelu(self.upconv2(fea1))
#         fea3 = self.lrelu(self.upconv3(torch.cat((fea1, fea2), 1)))
#         out = self.conv_last(self.lrelu(self.HRconv(fea3)))
# #         out = self.conv_last(self.lrelu(self.HRconv(fea2)))

#         return out


# RRDB net (scale 4)
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(nf+nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea1 = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea2 = self.lrelu(self.upconv2(F.interpolate(fea1, scale_factor=2, mode='nearest')))
        fea3 = self.lrelu(self.upconv3(fea2))
        fea4 = self.lrelu(self.upconv4(torch.cat((fea2, fea3), 1)))
        
        out = self.conv_last(self.lrelu(self.HRconv(fea4)))

        return out
    
# load model    
Genmodel_path = "./models/RRDB_ESRGAN_x4_png_l.pth"
Genmodel = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
Genmodel.load_state_dict(torch.load(Genmodel_path, map_location="cpu")["model_state_dict"])


# load test data
test_df = pd.read_json("./test_data.json", orient="records")
test_df.head()


# create dataset
class LRDataset(Dataset):
    def __init__(self, root, img_list, transform=None):
        super(LRDataset, self).__init__()
        self.root = root
        self.img_list = img_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_list[idx])
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)
        
        if self.transform is not None:
            lr_aug = self.transform(image=img)
            lr_img = lr_aug['image']
        
        return lr_img, self.img_list[idx]
    

# create dataloader
test_transform = A.Compose([ToTensor()])

test_img_list = list(test_df['image'])
test_dataset = LRDataset(root="/home/doriskao/project/super_resolution/test_data", img_list=test_img_list, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# inference and save upscale images
# save test images(x4)
i = 0
Genmodel.cpu()
Genmodel.eval()
for lrimgs, imgname in test_dataloader:
    with torch.no_grad():
        outimg = Genmodel(lrimgs)

    lrimgs = lrimgs.squeeze().permute(1, 2, 0).numpy()
    outimg = outimg.squeeze().permute(1, 2, 0).numpy()
    outimg = cv2.GaussianBlur(outimg, ksize=(0, 0), sigmaX=1.414/2, borderType=cv2.BORDER_REFLECT)
    outimg = np.clip(outimg, 0, 1)*255
    out_imgname = os.path.splitext(imgname[0])[0]
    cv2.imwrite(f"./test_data_scale4/{out_imgname}x4.png", cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR))