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


init_seed = 10
torch.manual_seed(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
random.seed(init_seed)

if __name__ == "__main__":
    feature_data_path = 'test_feature_TF_TDF_CST_MV_MFCC_60Hz_cut_zero'  # 提取的特征和标签文件夹
    fold_path = feature_data_path
    feature_path = os.path.join(fold_path, 'feature')
    # 单时频特征模型
    model_folder = r'E:\sdmurmur\ssdHeartMurmur\TF_ODConv_k3_weight_2_2_6\feature_TF_TDF_cut_zero'  # 存储模型的文件夹
    # 时频域特征+时域特征模型
    # model_folder = r'E:\sdmurmur\ssdHeartMurmur\TF_MFCC_TDFMVCST_ODC_k3_MM_FCCat133_withoutMFCC\feature_TF_TDF_CST_MV_MFCC_60Hz_cut_zero'  # 存储模型的文件夹
    model = AudioClassifierODconv()
    kfold = 5
    test_batch_size = 64
    for j in range(kfold):

        label_path = os.path.join(fold_path, 'label')

        test_data = np.load(feature_path + r'\test_loggamma.npy', allow_pickle=True)
        test_label = np.load(label_path + r'\test_label.npy', allow_pickle=True)
        test_location = np.load(label_path + r'\test_location.npy', allow_pickle=True)
        test_id = np.load(label_path + r'\test_id.npy', allow_pickle=True)
        test_index = np.load(label_path + r'\test_index.npy', allow_pickle=True)

        test_set = NewDataset(wav_label=test_label, wav_data=test_data, wav_index=test_index)
        test_loader = DataLoader(test_set, batch_size=test_batch_size)
        print("DataLoader is OK")
        fold = str(j) + r'_fold\model'  # 测试第i折模型
        model_path = os.path.join(model_folder, fold)
        # 加载模型
        model = torch.load(os.path.join(model_path, 'last_model'))
        # 采用最后一轮的模型进行评估
        model_result_path = os.path.join('test_result_odconv_k3_repeat_weight_2_2_6_last_model', fold_path, str(j) + '_fold')
        # model_result_path = os.path.join('test_result_odconv_k3_tf_tdf_MM_Fcat133', fold_path, str(j)+'_fold')
        # 设置环境变量，指定可见的 GPU 设备
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # 检查是否有可用的 GPU，并选择合适的计算设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # 放到设备中
        # 设置损失函数
        # weight = torch.tensor([1, 1, 1]).to(device)
        weight = torch.tensor([0.25, 0.25, 0.50]).to(device)  # sd 改变权重值，增加loud权重
        # criterion = Focal_Loss(gamma=2.5, weight=weight)
        criterion = Focal_Loss(gamma=2.5, weight=weight)  # sd 增大gamma

        print(model_path)
        # evaluate model,将模型设置为评估模式
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_test_loss = []
        all_test_acc = []
        all_y_pred = []  # 存放3s样本的预测概率
        all_y_pred_label = []  # 存放3s样本的真实标签
        all_label = []  # 存放3s样本的预测标签
        all_id = []
        all_location = []

        with torch.no_grad():
            # model.eval()
            for i, data in enumerate(test_loader):
                x, y, z = data
                x = x.to(device)
                # x = x.view(-1, 64 * 239).to(device)  # sd KAN
                y = y.to(device)
                z = z.to(device)

                outputs = model(x)
                loss = criterion(outputs, y.long())
                val_loss += loss.item()
                _, y_pred = outputs.max(1)
                num_correct = (y_pred == y).sum().item()
                acc = num_correct / test_batch_size
                val_acc += acc
                softmax = nn.Softmax(dim=1)
                all_y_pred.append(softmax(outputs).cpu().detach())
                all_label.append(y.cpu().detach())
                all_y_pred_label.append(y_pred.cpu().detach())
                # for ii in range(test_batch_size):
                #     all_id.append(test_id[z[ii].cpu().detach()])
                #     all_location.append(test_location[z[ii].cpu().detach()])  # 对应ID的所有片段听诊区位置
        all_y_pred = np.vstack(all_y_pred)  # 三种输出结果
        all_label = np.hstack(all_label)
        all_y_pred_label = np.hstack(all_y_pred_label)

        all_test_acc.append(val_acc / len(test_loader))
        all_test_loss.append(val_loss / len(test_loader))

        acc_metric = val_acc / len(test_loader)
        loss_metric = val_loss / len(test_loader)

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
        PCG_acc_soft_aver = (acc_metric + Soft_recall) / 2  # 准确率和soft找回率均值
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
        plt.savefig(result_path + '/PCG Confusion matrix.png', dpi=600)
        plt.close()
        # 保存历史loss到txt文件
        np_val_acc = np.array(all_test_acc).reshape((len(all_test_acc), 1))  # reshape是为了能够跟别的信息组成矩阵一起存储
        np_val_loss = np.array(all_test_loss).reshape((len(all_test_loss), 1))
        np_out = np.concatenate([np_val_acc, np_val_loss], axis=1)
        f = result_path + "/save_result.txt"
        mytime = datetime.now()
        with open(f, "a") as file:
            file.write("===============================================================================" + "\n")
            file.write(str(mytime) + "\n")
            file.write("-----------------PCG_vali_recall----------------- " + "\n")
            file.write("Absent: " + str('{:.4f}'.format(Absent_recall))
                       + "  Soft: " + str('{:.4f}'.format(Soft_recall))
                       + "  Loud: " + str('{:.4f}'.format(Loud_recall))
                       + "  PCG_UAR: " + str('{:.4f}'.format(PCG_UAR))
                       + "\n")
            file.write("-------------------PCG_vali_F1------------------- " + "\n")
            file.write("Absent: " + str('{:.4f}'.format(Absent_f1))
                       + "  Soft: " + str('{:.4f}'.format(Soft_f1))
                       + "  Loud: " + str('{:.4f}'.format(Loud_f1))
                       + "  UAF: " + str('{:.4f}'.format(UAF))
                       + "\n")
            # file.write('train_acc    val_acc   train_loss    val_loss' + "\n")
            # for i in range(len(np_out)):
            #     file.write(str(np_out[i]) + '\n')
        print("save result successful!!!")

    # checkpoint = torch.load(os.path.join(model_path, 'sd_last_model.pth'))
    # learning_rate = 0.005
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)
    # # 加载模型状态字典
    # model.load_state_dict(checkpoint['model_state_dict'])
    # # 加载优化器状态字典
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # # 打印模型的参数
    # print("Model's parameters:")
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}")
    #     print(f"Parameter shape: {param.shape}")
    #     print(f"Parameter values:\n{param}")
