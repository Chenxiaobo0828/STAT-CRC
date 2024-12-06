import os
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
import torch.optim as optim
from Dataset import MyDataSet
from torchvision import transforms
from torch.cuda.amp import autocast as autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Model import swin_tiny_patch4_window7_224 as create_model
from utils_II_CRC import read_split_data, train_one_epoch, evaluate, make_test_data, create_lr_scheduler
warnings.filterwarnings("ignore")

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    (train_images_path, train_images_fustat, train_images_futime,
     val_images_path, val_images_fustat, val_images_futime) = read_split_data(args.data_path,
                                                                              args.seed,
                                                                              args.ALL_cohort,
                                                                              args.Ex_val,
                                                                              args.timedata)

    test_images_path, test_images_fustat, test_images_futime = make_test_data(args.data_path,
                                                                              args.Ex_val,
                                                                              args.timedata)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224)),
                                     transforms.ColorJitter(brightness=0.25, contrast=0.25),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                     ]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((224, 224)),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_fustat = train_images_fustat,
                              images_futime = train_images_futime,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_fustat=val_images_fustat,
                            images_futime=val_images_futime,
                            transform=data_transform["val"])

    test_dataset = MyDataSet(images_path=test_images_path,
                             images_fustat=test_images_fustat,
                             images_futime=test_images_futime,
                             transform=data_transform["val"])

    for bb in range(100):
        bb = bb + 132
        if min(len(train_dataset) % bb, len(val_dataset) % bb, len(test_dataset) % bb) > 50:
            break
    args.batch_size = bb
    batch_size = args.batch_size
    print('batch_size=',batch_size, ' 数据集余数:', len(train_dataset) % bb, len(val_dataset) % bb, len(test_dataset) % bb)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs/4, eta_min=0.0000001, last_epoch=-1)

    train_cidx = 0; val_loss = 0; val_cidx = 0; test_loss = 0; test_cidx = 0; Test_cidx = 0; Best_epoch = 0
    train_cidx_list = []
    val_cidx_list = []
    test_cidx_list = []
    for epoch in range(args.epochs):
        train_loss, train_cidx = train_one_epoch(model=model,
                                                 optimizer=optimizer,
                                                 data_loader=train_loader,
                                                 device=device,
                                                 epoch=epoch,
                                                 scaler=scaler,
                                                 scheduler=scheduler,
                                                 task=args.task)

        val_loss, val_cidx = evaluate(model=model,
                                      data_loader=val_loader,
                                      device=device,
                                      epoch=epoch,
                                      part="valid",
                                      task=args.task)

        test_loss, test_cidx = evaluate(model=model,
                                        data_loader=test_loader,
                                        device=device,
                                        epoch=epoch,
                                        part="test",
                                        task=args.task)
        if args.task == "cox":
            tags = ["train_loss", "train_cidx", "val_loss", "val_cidx", "test_loss", "test_cidx", "learning_rate"]
        else:
            tags = ["train_loss", "train_auc", "val_loss", "val_auc", "test_loss", "test_auc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_cidx, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_cidx, epoch)
        tb_writer.add_scalar(tags[4], test_loss, epoch)
        tb_writer.add_scalar(tags[5], test_cidx, epoch)
        tb_writer.add_scalar(tags[6], optimizer.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(), "./weights/T2-20241123-seed{}-tr{}-va{}-te{}-model-{}.pth".format(args.seed, round(train_cidx, 3), round(val_cidx, 3), round(test_cidx, 3), epoch))

        if epoch % 20 == 0:
            print('之前的最佳结果, Best_epoch=', Best_epoch, Test_cidx)
    return train_loss, train_cidx, val_loss, val_cidx, test_loss, test_cidx


if __name__ == '__main__':
    for i in ['T2', 'DWI', 'DCE']:
        datalist = pd.DataFrame()
        task = 'cox'
        Dir = os.path.join('/home/dell/桌面/一步到位/RC_Prog/Crop_IMG/', i)
        timedata = pd.read_csv(Dir + "/" + "Clinical_File.csv", index_col=0, encoding='gbk')
        num_class = 1
        timedata = timedata.iloc[:, :2]

        ALL_cohort = ['Train_Cohort']
        EX_cohort = [['Val_Cohort']]

        for seed in [1]:
            for epoch in [60]:
                for lr in [4e-6]:
                    for Ex_val in EX_cohort:
                        print('------------------ seed = ', seed, '------------------')
                        print('------------------ lr = ', lr, '------------------')
                        print('------------------ Ex_val = ', Ex_val, '------------------')
                        parser = argparse.ArgumentParser()
                        parser.add_argument('--num_classes', type=int, default=num_class)
                        parser.add_argument('--epochs', type=int, default=epoch)
                        parser.add_argument('--batch-size', type=int, default=126)
                        parser.add_argument('--lr', type=float, default=lr)
                        parser.add_argument('--seed', type=float, default=seed)
                        parser.add_argument('--ALL_cohort', type=float, default=ALL_cohort)
                        parser.add_argument('--Ex_val', type=float, default=Ex_val)
                        parser.add_argument('--timedata', type=float, default=timedata)
                        parser.add_argument('--data-path', type=str, default=Dir)

                        # 预训练权重路径，如果不想载入就设置为空字符
                        parser.add_argument('--weights', type=str,
                                            default=r"weights/swin_tiny_patch4_window7_224.pth",
                                            help='initial weights path')
                        # 是否冻结权重
                        parser.add_argument('--freeze-layers', type=bool, default=False)
                        parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
                        parser.add_argument('--task', type=str, default=task)

                        opt = parser.parse_args()
                        train_loss, train_cidx, val_loss, val_cidx, test_loss, test_cidx = main(opt)
                        newlist = pd.DataFrame({"lr": [lr],
                                                "Ex_val": [Ex_val],
                                                "seed": [seed],
                                                "train_loss": [train_loss],
                                                "train_cidx": [train_cidx],
                                                "val_loss": [val_loss],
                                                "val_cidx": [val_cidx],
                                                "test_loss": [test_loss],
                                                "test_cidx": [test_cidx],
                                                })

                        datalist = pd.concat([datalist, newlist], axis=0)
                        datalist.to_csv(Dir + "/" + "results.csv", encoding='gbk', index=False)
