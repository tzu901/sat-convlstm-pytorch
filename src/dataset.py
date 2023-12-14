
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import List
import numpy as np
import os
import rasterio
import torch
from torchvision import transforms

def default_loader(path):
        return rasterio.open(path).read()

class DigitDataset(Dataset):

    # data loading
    def __init__(self, file_path,INPUT_FRAMES , OUTPUT_FRAMES,
                 loader=default_loader, transform=None
                 ):
        
        with open(file_path, 'r') as fh:
            self.X = []
            for line in fh:
                line = line.strip('\n').rstrip()
                self.X.append(line)


        # print(self.X, self.Y)
        self.loader = loader
        self.INPUT_FRAMES = INPUT_FRAMES
        self.OUTPUT_FRAMES = OUTPUT_FRAMES
        self.clip_count = INPUT_FRAMES + OUTPUT_FRAMES
        self.transform = transform


    # working for indexing
    def __getitem__(self, index):
        clip = self.X[index * self.clip_count: (index + 1) * self.clip_count]
        
        input = [self.loader(img_path) for img_path in clip[:self.INPUT_FRAMES]]
        output = [self.loader(img_path) for img_path in clip[self.INPUT_FRAMES:self.clip_count]]

        input_stack = np.stack(input)
        output_stack = np.stack(output)

        
        input_tensor = torch.from_numpy(input_stack).float().contiguous()
        output_tensor = torch.from_numpy(output_stack).float().contiguous()


        return index, output_tensor, input_tensor
        

    # return the length of our dataset
    def __len__(self):
        return len(self.X) // self.clip_count