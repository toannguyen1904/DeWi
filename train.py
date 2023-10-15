#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from utils.set_seeds import seed_everything
from utils.read_dataset import read_dataset
from utils.train_model import train
from config import seed, batch_size, root, checkpoint_path, init_lr, lr_decay_rate,\
    lr_milestones, weight_decay, end_epoch, dataset_path, input_size
from utils.auto_load_resume import auto_load_resume
import os
import argparse
from pytorch_metric_learning import losses, miners

from models.dewi import dewi_resnet50, dewi_resnet101, dewi_resnet152, dewi_resnext50_32x4d, dewi_resnext101_32x8d, dewi_resnext101_64x4d,\
    dewi_wide_resnet50_2, dewi_wide_resnet101_2


device = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="chosen model")
args = vars(ap.parse_args())


model_pool = {
    'dewi_resnet50': dewi_resnet50,
    'dewi_resnet101': dewi_resnet101,
    'dewi_resnet152': dewi_resnet152,
    'dewi_resnext50_32x4d': dewi_resnext50_32x4d,
    'dewi_resnext101_32x8d': dewi_resnext101_32x8d,
    'dewi_resnext101_64x4d': dewi_resnext101_64x4d,
    'dewi_wide_resnet50_2': dewi_wide_resnet50_2,
    'dewi_wide_resnet101_2': dewi_wide_resnet101_2,
}

pretrained_url_pool = dict.fromkeys(['dewi_resnet50'], "https://download.pytorch.org/models/resnet50-11ad3fa6.pth")
pretrained_url_pool.update(dict.fromkeys(['dewi_resnet101'], "https://download.pytorch.org/models/resnet101-cd907fc2.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnet152'], "https://download.pytorch.org/models/resnet152-f82ba261.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnext50_32x4d'], "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnext101_32x8d'], "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_resnext101_64x4d'], "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_wide_resnet50_2'], "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth"))
pretrained_url_pool.update(dict.fromkeys(['dewi_wide_resnet101_2'], "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth"))


def main():
    # set all the necessary seeds
    seed_everything(seed)
    # Read the dataset
    trainloader, valloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)

    model = model_pool.get(args["model"])(pth_url=pretrained_url_pool.get(args["model"]), pretrained=True)
    
    # define the CE loss function
    criterion = nn.CrossEntropyLoss()

    metric_loss = losses.TripletMarginLoss(0.2)
    miner = miners.BatchHardMiner()
    parameters = model.parameters()

    # define the optimizer
    optimizer = torch.optim.SGD(parameters, lr=init_lr, momentum=0.9, weight_decay=weight_decay)
    # define the learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate, verbose=True)

    # loading checkpoint
    save_path = os.path.join(checkpoint_path, args["model"])
    if os.path.exists(save_path):
        start_epoch, best_val_acc = auto_load_resume(model, optimizer, scheduler, save_path, status='train', device=device)
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        best_val_acc = 0.0
        start_epoch = 0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    

    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))
     # Train the model
    train(model=model,
          device=device,
          trainloader=trainloader,
          valloader=valloader,
          testloader=testloader,
          metric_loss=metric_loss,
          miner=miner,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          best_val_acc = best_val_acc)


if __name__ == '__main__':
    main()