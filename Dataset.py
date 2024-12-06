from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io
from skimage import io, exposure, img_as_uint, img_as_float
import torchvision
import SimpleITK

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_fustat: list,images_futime: list,transform=None):
        self.images_path = images_path
        self.images_fustat = images_fustat
        self.images_futime = images_futime
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = SimpleITK.ReadImage(self.images_path[item])
        img = SimpleITK.GetArrayFromImage(img)
        img = img.transpose(2,1,0)
        img = img.astype(np.float32)
        img = exposure.rescale_intensity(img, out_range="float32")
        img = self.transform(img)
        fustat = self.images_fustat[item]
        futime = self.images_futime[item]
        return self.images_path[item], img, fustat,futime

    @staticmethod
    def collate_fn(batch):
        images_path, images, fustats, futimes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        fustats = torch.as_tensor(fustats)
        futimes = torch.as_tensor(futimes)
        return images_path, images, fustats, futimes
