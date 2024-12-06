import os

import sys
import json
import pickle
import random
from typing import re

import pandas as pd
import torch
from tqdm import tqdm

import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from Model import coxloss, c_index
import math


def read_split_data(root: str, seed:int, ALL_cohort, Ex_val, timedata, val_rate: float = 0.2):
    random.seed(seed)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_images_path = []; train_images_fustat = []; train_images_futime = [];
    val_images_path = []; val_images_fustat = []; val_images_futime = [];

    print(set(ALL_cohort), set(Ex_val))
    print('------- model_cohort', list(set(ALL_cohort) - set(Ex_val)), '-------')
    for model_cohort in list(set(ALL_cohort) - set(Ex_val)):
        root2 = os.path.join(root, str(model_cohort))
        images = [os.path.join(root2, i) for i in os.listdir(root2)]
        # print(images)
        images = [name.replace('Smax+0', 'Smax') for name in images]
        images = [name.replace('Smax+1', 'Smax') for name in images]
        images = [name.replace('Smax-1', 'Smax') for name in images]
        images = list(set(images))
        val_path_initial = random.sample(images, k=int(len(images) * val_rate))
        val_path_1 = [name.replace('Smax', 'Smax+0') for name in val_path_initial]
        val_path_2 = [name.replace('Smax', 'Smax+1') for name in val_path_initial]
        val_path_3 = [name.replace('Smax', 'Smax-1') for name in val_path_initial]
        images_1 = [name.replace('Smax', 'Smax+0') for name in images]
        images_2 = [name.replace('Smax', 'Smax+1') for name in images]
        images_3 = [name.replace('Smax', 'Smax-1') for name in images]
        val_path = val_path_1 + val_path_2 + val_path_3
        images_else = images_1 + images_2 + images_3
        images_finally = [item for item in images_else if item not in val_path]
        for img_path in val_path:
            name = img_path.split('/')[-1]
            val_images_path.append(os.path.join(img_path))
            val_images_fustat.append(timedata.loc[name, "fustat"])
            val_images_futime.append(timedata.loc[name, "futime"])
        for img_path in images_finally:
            name = img_path.split('/')[-1]
            train_images_path.append(os.path.join(img_path))
            train_images_fustat.append(timedata.loc[name, "fustat"])
            train_images_futime.append(timedata.loc[name, "futime"])
            if timedata.loc[name, "fustat"] == 1:
                train_images_path.append(os.path.join(img_path))
                train_images_fustat.append(timedata.loc[name, "fustat"])
                train_images_futime.append(timedata.loc[name, "futime"])

    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    Train_data = pd.DataFrame({'images': train_images_path, 'fustat': train_images_fustat, 'futime': train_images_futime})
    Train_data.to_csv(root + "/" + str(seed) + "Train_results.csv", encoding='gbk', index=False)

    Val_data = pd.DataFrame({'images': val_images_path, 'fustat': val_images_fustat, 'futime': val_images_futime})
    Val_data.to_csv(root + "/" + str(seed) + "val_results.csv", encoding='gbk', index=False)

    return train_images_path, train_images_fustat, train_images_futime, val_images_path, val_images_fustat, val_images_futime



def make_test_data(root, Ex_val, timedata):
    test_images_path = []
    test_images_fustat = []
    test_images_futime = []
    for part in Ex_val:
        root2 = os.path.join(root, part)
        images = [os.path.join(root2, i) for i in os.listdir(root2)]
        for img_path in images:
            if True:
                name = img_path.split('/')[-1]
                test_images_path.append(os.path.join(img_path))
                test_images_fustat.append(timedata.loc[name, "fustat"])
                test_images_futime.append(timedata.loc[name, "futime"])

    print("{} images for testing.".format(len(test_images_path)))
    assert len(test_images_path) > 0, "number of testning images must greater than 0."

    Test_data = pd.DataFrame({'images': test_images_path, 'fustat': test_images_fustat, 'futime': test_images_futime})
    print('Test', root + "/" + "test_results.csv")
    Test_data.to_csv(root + "/" + "test_results.csv", encoding='gbk', index=False)

    return test_images_path, test_images_fustat,test_images_futime




def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, scheduler, task):
    model.train()
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)

    Cidxfunction = c_index(); sum_cidx = torch.zeros(1).to(device)
    lossfunction = coxloss(); sum_loss = torch.zeros(1).to(device)
    criterion = nn.CrossEntropyLoss()
    list_data = pd.DataFrame(columns=["images_path", "risk_pred", 'fustats', 'futimes'])
    for step, data in enumerate(data_loader):
        images_path, images, fustats, futimes = data
        with autocast():
            risk_pred = model(images.to(device))
        list_newdata = pd.DataFrame({"images_path": images_path,
                                     "risk_pred0": risk_pred.cpu().detach().numpy()[:, 0],
                                     # "risk_pred1": risk_pred[:,1].cpu().detach().numpy(),
                                     "fustat": fustats,
                                     "futime": futimes})
        list_data = list_data._append(list_newdata)

        risk_pred0 = torch.tensor(list_data['risk_pred0'].astype(float).values, dtype=torch.float32)
        # risk_pred1 = torch.tensor(list_data['risk_pred1'].astype(float).values, dtype=torch.float32)
        fustat = torch.tensor(list_data['fustat'].astype(float).values, dtype=torch.long)
        futime = torch.tensor(list_data['futime'].astype(float).values, dtype=torch.float32)

        if task == "cox":
            sum_cidx = Cidxfunction(risk_pred0, futime, fustat)
            loss = lossfunction(risk_pred.to(device), futimes.to(device), fustats.to(device), model)
            re = "Cindex"
        else:
            re = "AUC"
            sum_cidx = roc_auc_score(fustat, risk_pred1)
            loss = criterion(risk_pred.to(device), fustats.to(device, dtype=torch.float32))
        sum_loss += loss
        scaler.scale(loss).backward()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, {}: {:.3f}, lr*10^3: {:.5f}".format(epoch,
                                                                              sum_loss.item() / (step + 1),
                                                                              re,
                                                                              sum_cidx,
                                                                              optimizer.param_groups[0]["lr"]*1000)
        if not torch.isfinite(loss):
            break
            print('WARNING: non-finite loss, ending training ', loss); sys.exit(1)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
    return sum_loss.item(), sum_cidx.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, part, task):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)

    Cidxfunction = c_index(); sum_cidx = torch.zeros(1).to(device)
    lossfunction = coxloss(); sum_loss = torch.zeros(1).to(device)
    criterion = nn.CrossEntropyLoss()
    list_data = pd.DataFrame(columns=["images_path", "risk_pred", 'fustats', 'futimes'])
    for step, data in enumerate(data_loader):
        images_path, images, fustats, futimes = data
        with autocast():
            risk_pred = model(images.to(device))
        list_newdata = pd.DataFrame({"images_path": images_path,
                                     "risk_pred0": risk_pred[:,0].cpu().detach().numpy(),
                                     # "risk_pred1": risk_pred[:,1].cpu().detach().numpy(),
                                     "fustats": fustats,
                                     "futimes": futimes})
        list_data = list_data._append(list_newdata)

        risk_pred0 = torch.tensor(list_data['risk_pred0'].astype(float).values, dtype=torch.float32)
        # risk_pred1 = torch.tensor(list_data['risk_pred1'].astype(float).values, dtype=torch.float32)
        fustats = torch.tensor(list_data['fustats'].astype(float).values, dtype=torch.long)
        futimes = torch.tensor(list_data['futimes'].astype(float).values, dtype=torch.float32)
        if task == "cox":
            re = "Cindex"
            sum_loss = lossfunction(risk_pred0, futimes, fustats, model)
            sum_cidx = Cidxfunction(risk_pred0, futimes, fustats)
        else:
            re = "AUC"
            sum_loss = criterion(torch.stack([risk_pred0, risk_pred1], dim=1), fustats)
            sum_cidx = roc_auc_score(fustats, risk_pred1)

        data_loader.desc = "[{} epoch {}] loss: {:.3f}, {}: {:.3f}, lr*10^3: {:.5f}".format(part, epoch,
                                                                                            sum_loss*step,
                                                                                            re,
                                                                                            sum_cidx,
                                                                                            0)

    return sum_loss.item(), sum_cidx.item()


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
