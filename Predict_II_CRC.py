import os
import pandas as pd
import torch
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import SimpleITK
from Model import swin_tiny_patch4_window7_224 as create_model
from skimage import io, exposure, img_as_uint, img_as_float
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from dataset import MyDataSet
from utils_II_CRC import evaluate, make_test_data

Dir = r'/home/dell/桌面/II_CRC_Cox/II_CRC_data'
task = "cox"
timedata = pd.read_csv(Dir + "/" + "Clinical_File.csv", index_col=0, encoding='gbk')
list_data = pd.DataFrame(columns=["ID", "Probability"])
timedata = timedata.iloc[:, :2]


EX_cohort = ['Predict_Cohort']


test_images_path, test_images_fustat, test_images_futime = make_test_data(Dir,
                                                                          EX_cohort,
                                                                          timedata)

data_transform = {
    "val": transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((224, 224)),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ])}

test_dataset = MyDataSet(images_path=test_images_path,
                         images_fustat=test_images_fustat,
                         images_futime=test_images_futime,
                         transform=data_transform["val"])


test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=16,
                                          collate_fn=test_dataset.collate_fn)


model_name = 'Train.pth'
device = 'cuda:0'
model2 = create_model(num_classes=1).to(device)
model_weight_path = Dir + "/" + model_name
model2.load_state_dict(torch.load(model_weight_path, map_location=device))
model2.eval()

for step, data in enumerate(test_loader):
    images_path, images, fustats, futimes = data
    risk_pred = model2(torch.unsqueeze(images, dim=0)[0,:,:,:].to(device))[:, 0]
    for i in range(len(images_path)):
        list_newdata = pd.DataFrame({"ID": [images_path[i]], "futime": futimes[i].detach().numpy(),
                                     "fustats": fustats[i].detach().numpy(), "Probability": risk_pred[i].cpu().detach().numpy()})
        list_data = list_data.append(list_newdata)
list_data.to_csv(os.path.join(Dir, "Predict_Outcome.csv"), encoding="gbk", index=False)





