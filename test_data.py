import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from Imbanlance_Loss import Focal_Loss, DiceLoss, PolyLoss
import math
import torch.optim as optim
from CNN import (AudioClassifier, AudioClassifierFuseODconv, AudioClassifierODconv)
# from efficient_kan import KAN
from My_Dataloader import NewDataset, TrainDataset, Dataset2, MyDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from patient_information import get_locations, cal_patient_acc, single_result, location_result
import random

if __name__ == "__main__":
    feature_data_path = 'feature_TF_TDF_cut_zero'  # 提取的特征和标签文件夹
    fold_path = feature_data_path
    feature_path = os.path.join(fold_path, 'feature')
    model = AudioClassifierODconv()
    model_path = r'E:\sdmurmur\ssdHeartMurmur\TF_TDF_cut_zero_ODconv_k3_withoutfuse_best_loud\feature_TF_TDF_cut_zero\0_fold\model'
    checkpoint = torch.load(os.path.join(model_path, 'sd_last_model.pth'))
    learning_rate = 0.005

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)
    # 加载模型状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    # 加载优化器状态字典
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 打印模型的参数
    print("Model's parameters:")
    for name, param in model.named_parameters():
        print(f"Layer: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter values:\n{param}")
