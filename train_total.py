import os
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
# from datetime import datetime
from Imbanlance_Loss import Focal_Loss, DiceLoss, PolyLoss
# import math
# import torch.optim as optim
from CNN import (AudioClassifier, AudioClassifierFuseODconv, AudioClassifierODconv)
from AMG import AmgModel, resblock
# from efficient_kan import KAN
from My_Dataloader import NewDataset, TrainDataset, Dataset2, MyDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
# from patient_information import get_locations, cal_patient_acc, single_result, location_result
import random
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix
from datetime import datetime

init_seed = 12
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
# sd 2025/04/04 使用4070s进行训练集和测试集的cut zero
if __name__ == "__main__":

    # 提取的特征和标签文件夹
    train_feature_data_path = r"D:\sdmurmur\sdMurmurFiles\feature\train_vali_data_cz_TF_TDF_MV"
    test_feature_data_path = r"D:\sdmurmur\sdMurmurFiles\feature\test_data_cz_TF_TDF_MV"
    # 提取的特征和标签文件夹
    # feature_data_path = r"E:\sdmurmur\ssdHeartMurmurFiles\calibrated_train_vali_new_mixed_data_feature\TF_log_mel_32_feature"
    # 剪切后的数据文件夹
    # cut_data_kfold = r'data_kfold_out'
    cut_data_kfold = r"D:\sdmurmur\sdMurmurFiles\calibrated_train_vali_dataset_cz"

    train_feature_path = os.path.join(train_feature_data_path, 'feature')
    train_label_path = os.path.join(train_feature_data_path, 'label')
    test_feature_path = os.path.join(test_feature_data_path, 'feature')
    test_label_path = os.path.join(test_feature_data_path, 'label')

    train_data = np.load(train_feature_path + r'\train_loggamma.npy', allow_pickle=True)  # 加载训练集和验证集数据，验证集和测试集搞反了，test_loggamma.npy应该存为vali_loggamma.npy
    test_data = np.load(test_feature_path + r'\test_loggamma.npy', allow_pickle=True)

    train_label = np.load(train_label_path + r'\train_label.npy', allow_pickle=True)  # 加载训练集和测试集标签
    test_label = np.load(test_label_path + r'\test_label.npy', allow_pickle=True)

    test_location = np.load(test_label_path + r'\test_location.npy', allow_pickle=True)
    test_id = np.load(test_label_path + r'\test_id.npy', allow_pickle=True)
    test_index = np.load(test_label_path + r'\test_index.npy', allow_pickle=True)

    train_set = TrainDataset(wav_label=train_label, wav_data=train_data)  # 将训练集和测试集转换成torch张量
    test_set = NewDataset(wav_label=test_label, wav_data=test_data, wav_index=test_index)

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

    # ========================/ dataloader /========================== #
    # DataLoader输入的dataset应该实现__len__()和__getitem__()方法，分别返回数据集的长度和获取单个样本的方法
    train_loader = DataLoader(train_set, batch_size=train_batch_size,
                              sampler=weighted_sampler, drop_last=True)  # sampler主要用于定义数据加载的顺序和方式
    test_loader = DataLoader(test_set, batch_size=test_batch_size)
    print("DataLoader is OK")
    # 模型选择
    model = AudioClassifierFuseODconv()  # sd Fuse ODconv gamma=2.5
    # model = AudioClassifierODconv()
    # model = AmgModel(resblock, 1, 3)
    # model = AudioClassifier()
    # model_result_path = r"E:\sdmurmur\ssdHeartMurmurFiles\train_vali_new_results\train_vali_new_mixed\TF_TDF_ODC_1_1_1_12"
    # model_result_path = (r"E:\sdmurmur\ssdHeartMurmurFiles\train_vali_new_results"
    #                      r"\train_vali_new_mixed\TF_SK_105_1_12_12")
    model_result_path = r"D:\sdmurmur\sdMurmurFiles\result\ODC_TF_TDF_MV"
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
    weight = torch.tensor([1, 1, 1]).to(device)
    # weight = torch.tensor([1.05, 1, 1.2]).to(device)  # sd 改变权重值，增加loud权重
    criterion = Focal_Loss(gamma=2.5, weight=weight)
    # 创建交叉熵损失函数
    # criterion = nn.CrossEntropyLoss()  # AMG模型损失函数

    # 保存验证集准确率最大时的模型
    model_path = os.path.join(model_result_path, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # result_path = os.path.join(model_result_path, "ResultFile")
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)

    # train model
    no_better_epoch = 0
    torch.manual_seed(10)


    # 初始化存储 Loss 的列表
    all_train_loss = []  # 添加在训练循环前
    all_test_loss = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        all_y_pred = []
        all_y_true = []
        model.train()  # 设为训练模式
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

        # 记录平均 Loss
        avg_train_loss = train_loss / len(train_loader)
        all_train_loss.append(avg_train_loss)  # 添加到列表

        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        # model.eval()
        test_loss = 0.0
        test_acc = 0.0
        all_test_acc = []
        all_y_pred = []  # 存放3s样本的预测概率
        all_y_pred_label = []  # 存放3s样本的真实标签
        all_label = []  # 存放3s样本的预测标签
        all_id = []
        all_location = []
        # 初始化评估指标
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(test_loader):
                x, y, z = data
                x = x.to(device)
                # x = x.view(-1, 64 * 239).to(device)  # sd KAN
                y = y.to(device)
                z = z.to(device)

                outputs = model(x)
                loss = criterion(outputs, y.long())
                test_loss += loss.item()
                _, y_pred = outputs.max(1)
                num_correct = (y_pred == y).sum().item()
                acc = num_correct / test_batch_size
                test_acc += acc
                softmax = nn.Softmax(dim=1)
                all_y_pred.append(softmax(outputs).cpu().detach())
                all_label.append(y.cpu().detach())
                all_y_pred_label.append(y_pred.cpu().detach())

                predictions.extend(y_pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

            # 计算每一类的召回率和 F1 分数
            recall_per_class = recall_score(labels, predictions, average=None)
            f1_per_class = f1_score(labels, predictions, average=None)
            # 打印每一类的召回率和 F1 分数
            class_names = ['Absent 0', 'Soft 1', 'Loud 2']  # 假设有三类
            for i, class_name in enumerate(class_names):
                print(f'{class_name} - Recall: {recall_per_class[i]:.4f}, F1 Score: {f1_per_class[i]:.4f}')

            all_y_pred = np.vstack(all_y_pred)  # 三种输出结果
            all_label = np.hstack(all_label)
            all_y_pred_label = np.hstack(all_y_pred_label)

            all_test_acc.append(test_acc / len(test_loader))
            all_test_loss.append(test_loss / len(test_loader))

            # test set 结果统计，PCG分类性能
            # 将预测标签和真实标签转换为numpy数组
            y_pred = np.array(all_y_pred_label)
            y_true = np.array(all_label)
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            # 计算召回率 F1
            Absent_num = np.sum(cm[0])
            Soft_num = np.sum(cm[1])
            Loud_num = np.sum(cm[2])
            Absent_recall = cm[0][0] / Absent_num
            Soft_recall = cm[1][1] / Soft_num
            Loud_recall = cm[2][2] / Loud_num

            PCG_UAR = (Absent_recall + Soft_recall + Loud_recall) / 3

            print("------------------------------PCG result------------------------------")
            print("Absent_recall: %.4f, Soft_recall: %.4f, Loud_recall: %.4f,PCG_UAR: %.4f"
                  % (Absent_recall, Soft_recall, Loud_recall, PCG_UAR))
            a = np.sum(cm, 0)
            Absent_Precision = cm[0][0] / a[0]
            Soft_Precision = cm[1][1] / a[1]
            Loud_Precision = cm[2][2] / a[2]

            Absent_f1 = (2 * Absent_recall * Absent_Precision) / (Absent_recall + Absent_Precision)
            Soft_f1 = (2 * Soft_recall * Soft_Precision) / (Soft_recall + Soft_Precision)
            Loud_f1 = (2 * Loud_recall * Loud_Precision) / (Loud_recall + Loud_Precision)
            PCG_f1 = (Absent_f1 + Soft_f1 + Loud_f1) / 3
            print("Absent_F1: %.4f, Soft_F1: %.4f, Loud_F1: %.4f, PCG_F1: %.4f"
                  % (Absent_f1, Soft_f1, Loud_f1, PCG_f1))
            result_path = os.path.join(model_result_path, "ResultFile")
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            # 存储到.txt文件的数据
            UAF = (Absent_f1 + Soft_f1 + Loud_f1) / 3
            PCG_UAR = (Absent_recall + Soft_recall + Loud_recall) / 3

            # PCG混淆矩阵
            # 将预测标签和真实标签转换为numpy数组
            plt.figure()
            plt.imshow(cm, cmap=plt.cm.Blues)
            plt.colorbar()
            # 显示矩阵元素的数值
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha='center', va='center')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.xticks([0, 1, 2], ['absent', 'soft', 'loud'])
            plt.yticks([0, 1, 2], ['absent', 'soft', 'loud'])
            plt.title('Confusion matrix')
            plt.savefig(result_path + f'/Epoch {epoch + 1} PCG Confusion matrix.png', dpi=600)
            plt.close()
            # 保存历史loss到txt文件
            np_val_acc = np.array(all_test_acc).reshape((len(all_test_acc), 1))  # reshape是为了能够跟别的信息组成矩阵一起存储
            np_val_loss = np.array(all_test_loss).reshape((len(all_test_loss), 1))

            f = result_path + "/save_result.txt"
            mytime = datetime.now()
            with open(f, "a") as file:
                file.write(f"Epoch {epoch + 1} " + "\n")
                file.write(str(mytime) + "\n")
                file.write("-----------------PCG_test_recall----------------- " + "\n")
                file.write("Absent: " + str('{:.4f}'.format(Absent_recall))
                           + "  Soft: " + str('{:.4f}'.format(Soft_recall))
                           + "  Loud: " + str('{:.4f}'.format(Loud_recall))
                           + "  PCG_UAR: " + str('{:.4f}'.format(PCG_UAR))
                           + "\n")
                file.write("-------------------PCG_test_F1------------------- " + "\n")
                file.write("Absent: " + str('{:.4f}'.format(Absent_f1))
                           + "  Soft: " + str('{:.4f}'.format(Soft_f1))
                           + "  Loud: " + str('{:.4f}'.format(Loud_f1))
                           + "  UAF: " + str('{:.4f}'.format(UAF))
                           + "\n")
                # file.write('train_acc    val_acc   train_loss    val_loss' + "\n")
                # for i in range(len(np_out)):
                #     file.write(str(np_out[i]) + '\n')
            print("save result successful!!!")

    # 训练结束后绘制 Loss 曲线
    plt.figure()
    plt.plot(all_train_loss, label='Training Loss', color='blue')
    plt.plot(all_test_loss, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    # 保存图片
    loss_curve_path = os.path.join(model_result_path, "loss_curve.jpg")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Loss curve saved to: {loss_curve_path}')

    torch.save(
        model,
        os.path.join(model_path, 'last_model'),
    )
    print('train total completed')
