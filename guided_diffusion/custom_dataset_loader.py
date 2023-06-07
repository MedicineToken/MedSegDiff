import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from glob import glob
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):

        print("loading data from the directory :",data_path)
        path=data_path
        images = sorted(glob(os.path.join(path, "images/*.png")))
        masks = sorted(glob(os.path.join(path, "masks/*.png")))

        self.name_list = images[:2]
        self.label_list = masks[:2]
        self.data_path = path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.mode == 'Training':
            label = 0 if self.label_list[index] == 'benign' else 1
        else:
            label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        if self.mode == 'Training':
            return (img, mask, name)
        else:
            return (img, mask, name)