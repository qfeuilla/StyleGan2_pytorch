import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CreateDataset(Dataset):
    def __init__(self, path, transform):
        self.dir = path
        self.path = os.listdir(path)
        self.transform = transform
        self.dirLen = len(self.path)

    def __len__(self):
        return self.dirLen

    def __getitem__(self, index):
        
        rand = np.random.randint(0, self.dirLen)
        img = Image.open(self.dir + self.path[rand])
        img = self.transform(img)

        return img