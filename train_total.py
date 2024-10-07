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
from sklearn.metrics import recall_score, f1_score

init_seed = 10
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
random.seed(init_seed)
torch.backends.cudnn.deterministic = True


# sd 2024/07/24 推送
# sd 2024/09/08 add ODConv
# sd 2024/09/09 add ODConv concat model
# sd 2024/09/13 add cwt feature
# sd 2024/09/19 add dfm.py
# sd 2024/09/24 将时域信号重复后送入与时频域信号相同网络后在通道层面拼接
# sd 2024/09/28 添加数据分帧后的均值和方差作为特征
# sd 2024/09/30 添加MFCC特征，进行多模态（3）特征融合
# sd 2024/10/06 改变FocalLoss参数调整单时频域特征的结果，重跑特征拼接模型Fcat5  tdf_cat_sum
if __name__ == "__main__":
    # fold = '4_fold'  # 训练第i折
    feature_data_path = 'train_total_feature_TF_TDF_60Hz_cut_zero'  # 提取的特征和标签文件夹
    # cut_data_kfold = r'data_kfold_out'
    cut_data_kfold = r'data_kfold_cut_zero'

    feature_path = os.path.join(feature_data_path, 'feature')
    label_path = os.path.join(feature_data_path, 'label')

    train_data = np.load(feature_path + r'\train_loggamma.npy', allow_pickle=True)  # 加载训练集和验证集数据，验证集和测试集搞反了，test_loggamma.npy应该存为vali_loggamma.npy
    # vali_data = np.load(feature_path + r'\vali_loggamma.npy', allow_pickle=True)

    train_label = np.load(label_path + r'\train_label.npy', allow_pickle=True)  # 加载训练集和测试集标签
    # vali_label = np.load(label_path + r'\vali_label.npy', allow_pickle=True)

    # vali_location = np.load(label_path + r'\vali_location.npy', allow_pickle=True)
    # vali_id = np.load(label_path + r'\vali_id.npy', allow_pickle=True)
    # vali_index = np.load(label_path + r'\vali_index.npy', allow_pickle=True)

    train_set = TrainDataset(wav_label=train_label, wav_data=train_data)  # 将训练集和测试集转换成torch张量
    # vali_set = NewDataset(wav_label=vali_label, wav_data=vali_data, wav_index=vali_index)

    print((train_set.data[0]).shape)
    num_classes = 3
    class_count = np.zeros(num_classes, dtype=int)
    for _, label in train_set:
        class_count[label] += 1
    print("train_set:", 'absent:', class_count[0], 'soft:', class_count[1], 'loud:', class_count[2])
    test_class_count = np.zeros(num_classes, dtype=int)

    # 计算每个类别采样权重，解决数据不平衡问题
    target_count = max(class_count)
    class_weights = [target_count / count for count in class_count]
    # 创建权重采样器
    weights = [class_weights[label] for _, label in train_set]  # 类别占比低，权重高
    weighted_sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)

    # train_batch_size = 128
    train_batch_size = 128
    # test_batch_size = 1
    test_batch_size = 1
    learning_rate = 0.005
    # learning_rate = 0.002
    num_epochs = 20
    # num_epochs = 30  # sd Fuse
    # num_epochs = 60  # sd KAN 会过拟合
    img_size = (32, 240)
    patch_size = (8, 20)
    encoders = 1
    num_heads = 12

    # ========================/ dataloader /========================== #
    # DataLoader输入的dataset应该实现__len__()和__getitem__()方法，分别返回数据集的长度和获取单个样本的方法
    train_loader = DataLoader(train_set, batch_size=train_batch_size,
                              sampler=weighted_sampler, drop_last=True)  # sampler主要用于定义数据加载的顺序和方式
    # test_loader = DataLoader(vali_set, batch_size=test_batch_size)
    print("DataLoader is OK")
    # 模型选择
    model = AudioClassifierFuseODconv()  # sd Fuse ODconv gamma=2.5
    # model = AudioClassifier()
    model_result_path = os.path.join('train_total_TF_MFCC_TDFMVCST_ODC_k3_MM_FCCat133_withoutMFCC', feature_data_path)
    # model_result_path = os.path.join('Aweight_TimeFreq_result', fold_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 放到设备中
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)
    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 15, 20], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 15, 20, 25, 30], gamma=0.2)  # sd Fuse会过拟合

    # 设置损失函数
    # weight = torch.tensor([1, 1, 1]).to(device)
    weight = torch.tensor([0.25, 0.25, 0.50]).to(device)  # sd 改变权重值，增加loud权重
    # criterion = Focal_Loss(gamma=2.5, weight=weight)
    criterion = Focal_Loss(gamma=2.5, weight=weight)  # sd 增大gamma
    # criterion = nn.CrossEntropyLoss()  # sd KAN
    # 保存验证集准确率最大时的模型
    model_path = os.path.join(model_result_path, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_result_path, "ResultFile")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # train model
    no_better_epoch = 0
    torch.manual_seed(10)
    model.train()  # sd KAN
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        all_y_pred = []
        all_y_true = []
        for batch_idx, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            # x = x.view(-1, 64*239).to(device)  # sd KAN
            y = y.to(device)

            outputs = model(x)
            optimizer.zero_grad()
            loss = criterion(outputs, y.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, y_pred = outputs.max(1)  # 返回最大概率值和对应索引，y_pred即为取对应概率索引值：absent:0, soft:1, loud:2
            num_correct = (y_pred == y).sum().item()
            acc = num_correct / train_batch_size
            train_acc += acc
            # all_y_pred.append(y_pred.cpu().detach())
            all_y_pred.extend(y_pred.cpu().detach().numpy())
            all_y_true.extend(y.cpu().detach().numpy())
        scheduler.step()
        # 计算每个类别的召回率和 F1 分数
        recall_per_class = recall_score(all_y_true, all_y_pred, average=None)
        f1_per_class = f1_score(all_y_true, all_y_pred, average=None)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        # all_train_acc.append(train_acc / len(train_loader))
        # all_train_loss.append(train_loss / len(train_loader))
        class_names = ['Absent', 'Soft', 'Loud']
        for i, class_name in enumerate(class_names):
            print(
                f'Epoch {epoch + 1}, {class_name} - Recall: {recall_per_class[i]:.4f}, F1 Score: {f1_per_class[i]:.4f}')

        # 打印平均训练损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        print(
            f'Epoch {epoch + 1}, Average Train Loss: {avg_train_loss:.4f}, Average Train Accuracy: {avg_train_acc:.4f}')
        # 只在最后一个 epoch 记录结果
        if epoch == num_epochs - 1:
            # 打开一个文件用于写入结果
            with open('training_metrics_last_epoch.txt', 'w') as f:
                f.write(f'Epoch {epoch + 1}\n')
                for i, class_name in enumerate(class_names):
                    f.write(f'{class_name} - Recall: {recall_per_class[i]:.4f}, F1 Score: {f1_per_class[i]:.4f}\n')

                # 写入平均训练损失和准确率
                f.write(f'Average Train Loss: {avg_train_loss:.4f}, Average Train Accuracy: {avg_train_acc:.4f}\n\n')
    torch.save(
        model,
        os.path.join(model_path, 'last_model'),
    )
    print('train total completed')
