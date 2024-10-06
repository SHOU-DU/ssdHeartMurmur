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
    kfold = 5
    test_flag = False
    # 若使用测试集，则test_flag = True
    if test_flag:
        kfold = 0
    for i in range(kfold):
        fold = str(i) + '_fold'  # 训练第i折
        # fold = '2_fold'
        print(f'this is {fold}')

        # fold = '4_fold'  # 训练第i折
        feature_data_path = 'feature_TF_TDF_60Hz_cut_zero'  # 提取的特征和标签文件夹
        # cut_data_kfold = r'data_kfold_out'
        cut_data_kfold = r'data_kfold_cut_zero'
        if not test_flag:
            fold_path = os.path.join(feature_data_path, fold)
            cut_data = os.path.join(cut_data_kfold, fold, 'vali_data')
        else:
            fold_path = feature_data_path
            cut_data = cut_data_kfold
        feature_path = os.path.join(fold_path, 'feature')
        label_path = os.path.join(fold_path, 'label')

        train_data = np.load(feature_path + r'\train_loggamma.npy', allow_pickle=True)  # 加载训练集和验证集数据，验证集和测试集搞反了，test_loggamma.npy应该存为vali_loggamma.npy
        vali_data = np.load(feature_path + r'\vali_loggamma.npy', allow_pickle=True)

        train_label = np.load(label_path + r'\train_label.npy', allow_pickle=True)  # 加载训练集和测试集标签
        vali_label = np.load(label_path + r'\vali_label.npy', allow_pickle=True)

        vali_location = np.load(label_path + r'\vali_location.npy', allow_pickle=True)
        vali_id = np.load(label_path + r'\vali_id.npy', allow_pickle=True)
        vali_index = np.load(label_path + r'\vali_index.npy', allow_pickle=True)

        train_set = TrainDataset(wav_label=train_label, wav_data=train_data)  # 将训练集和测试集转换成torch张量
        vali_set = NewDataset(wav_label=vali_label, wav_data=vali_data, wav_index=vali_index)

        print((train_set.data[0]).shape)
        num_classes = 3
        class_count = np.zeros(num_classes, dtype=int)
        for _, label in train_set:
            class_count[label] += 1
        print("train_set:", 'absent:', class_count[0], 'soft:', class_count[1], 'loud:', class_count[2])
        test_class_count = np.zeros(num_classes, dtype=int)
        for data, label, index in vali_set:
            test_class_count[label] += 1
        print("vali_set:", 'absent:', test_class_count[0], 'soft:', test_class_count[1], 'loud:', test_class_count[2])

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
        test_loader = DataLoader(vali_set, batch_size=test_batch_size)
        print("DataLoader is OK")
        # 模型选择
        model = AudioClassifierFuseODconv()  # sd Fuse ODconv gamma=2.5
        # model = AudioClassifier()
        model_result_path = os.path.join('TF_TDF_60Hz_FCCat5_repeat', fold_path)
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
        # weight = torch.tensor([0.2, 0.2, 0.60]).to(device)  # sd 改变权重值，增加loud权重
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

        all_train_loss = []
        all_train_acc = []
        all_val_loss = []
        all_val_acc = []
        best_val_acc = -np.inf
        best_val_UAR = -np.inf
        best_val_F1 = -np.inf
        best_val_acc_soft = -np.inf
        best_val_soft = -np.inf

        aver_PCG_acc = []
        aver_PCG_UAR = []
        aver_PCG_absent = []
        aver_PCG_soft = []
        aver_PCG_loud = []

        best_val_patient_acc = -np.inf
        best_val_loss = 1
        best_train_acc = -np.inf

        # train model
        no_better_epoch = 0
        torch.manual_seed(10)
        model.train()  # sd KAN
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_acc = 0.0
            all_y_pred = []
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
                all_y_pred.append(y_pred.cpu().detach())
            scheduler.step()
            print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
            all_train_acc.append(train_acc / len(train_loader))
            all_train_loss.append(train_loss / len(train_loader))

            # evaluate model
            val_loss = 0.0
            val_acc = 0.0

            all_y_pred = []  # 存放3s样本的预测概率
            all_y_pred_label = []  # 存放3s样本的真实标签
            all_label = []  # 存放3s样本的预测标签
            all_id = []
            all_location = []
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
                    val_loss += loss.item()
                    _, y_pred = outputs.max(1)
                    num_correct = (y_pred == y).sum().item()
                    acc = num_correct / test_batch_size
                    val_acc += acc
                    softmax = nn.Softmax(dim=1)
                    all_y_pred.append(softmax(outputs).cpu().detach())
                    all_label.append(y.cpu().detach())
                    all_y_pred_label.append(y_pred.cpu().detach())
                    for ii in range(test_batch_size):
                        all_id.append(vali_id[z[ii].cpu().detach()])
                        all_location.append(vali_location[z[ii].cpu().detach()])  # 对应ID的所有片段听诊区位置

            all_y_pred = np.vstack(all_y_pred)  # 三种输出结果
            all_label = np.hstack(all_label)
            all_y_pred_label = np.hstack(all_y_pred_label)

            all_val_acc.append(val_acc / len(test_loader))
            all_val_loss.append(val_loss / len(test_loader))

            acc_metric = val_acc / len(test_loader)
            loss_metric = val_loss / len(test_loader)

            # 计算人头的准确率
            vali_data_directory = os.path.join(cut_data)
            # vali_data_directory：切割好的验证集，all_id：对应验证集ID, all_y_pred：三种输出结果，all_location：顺序存储的所有听诊区片段
            patient_pre, patient_ture_label, patient_acc = cal_patient_acc(vali_data_directory, all_id, all_y_pred,
                                                                           all_location)

            # if not os.path.exists(r'E:\sdmurmur\segmentation\TimeFreq_result\data_kfold_feature\\all_id'):
            #     os.makedirs('all_id')
            # np.save('all_id' + r'\all_id.txt', all_id)

            # ------结果统计------
            print(
                "======================================================================================================================")
            print(
                "Epoch: %d, Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, "
                "Val Acc: %.4f,patient Acc: %.4f "
                % (
                    epoch,
                    train_loss / len(train_loader),
                    train_acc / len(train_loader),
                    val_loss / len(test_loader),
                    val_acc / len(test_loader),
                    patient_acc

                )
            )

            # PCG分类性能
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

            # 患者分类性能
            # 将预测标签和真实标签转换为numpy数组
            y_pred = np.array(patient_pre)
            y_true = np.array(patient_ture_label)
            # 计算混淆矩阵
            cm1 = confusion_matrix(y_true, y_pred)
            # 计算召回率 F1
            Absent_num = np.sum(cm1[0])
            Soft_num = np.sum(cm1[1])
            Loud_num = np.sum(cm1[2])
            All_num = Absent_num + Soft_num + Loud_num
            Absent_recall_patient = cm1[0][0] / Absent_num
            Soft_recall_patient = cm1[1][1] / Soft_num
            Loud_recall_patient = cm1[2][2] / Loud_num
            Patient_UAR = (Absent_recall_patient + Soft_recall_patient + Loud_recall_patient) / 3
            Patient_WAR = (
                                      Absent_recall_patient * Absent_num + Soft_recall_patient * Soft_num + Loud_recall_patient * Loud_num) / All_num
            # Patient_WAcc = (Absent_num*cm1[0][0]+Soft_num*cm1[1][1]+Loud_num*cm1[2][2])/All_num
            print("------------------------------Patient result------------------------------")
            print("Absent_recall: %.4f, Soft_recall: %.4f, Loud_recall: %.4f, Patient_UAR: %.4f, Patient_WAR: %.4f"
                  % (Absent_recall_patient, Soft_recall_patient, Loud_recall_patient, Patient_UAR, Patient_WAR))
            a = np.sum(cm1, 0)
            Absent_Precision_patient = cm1[0][0] / a[0]
            Soft_Precision_patient = cm1[1][1] / a[1]
            Loud_Precision_patient = cm1[2][2] / a[2]

            Absent_f1_patient = (2 * Absent_recall_patient * Absent_Precision_patient) / (
                    Absent_recall_patient + Absent_Precision_patient)
            Soft_f1_patient = (2 * Soft_recall_patient * Soft_Precision_patient) / (
                    Soft_recall_patient + Soft_Precision_patient)
            Loud_f1_patient = (2 * Loud_recall_patient * Loud_Precision_patient) / (
                    Loud_recall_patient + Loud_Precision_patient)
            Patient_f1 = (Absent_f1_patient + Soft_f1_patient + Loud_f1_patient) / 3
            print("Absent_F1: %.4f, Soft_F1: %.4f, Loud_F1: %.4f,Patient_F1: %.4f"
                  % (Absent_f1_patient, Soft_f1_patient, Loud_f1_patient, Patient_f1))

            best_acc_metric = best_val_acc
            best_uar = best_val_UAR

            if epoch >= 15:
                # if acc_metric > best_acc_metric:
                if Patient_f1 > best_val_F1:
                    torch.save(
                        model,
                        os.path.join(model_path, 'best_model'),
                    )

                    # sd：保存模型和参数
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(model_path, 'sd_best_model.pth'))

                    print(
                        "Saving best_model model to:",
                        os.path.join(model_path, 'best_model'),
                    )
                    best_train_acc = train_acc / len(train_loader)
                    best_val_acc = acc_metric
                    best_val_patient_acc = patient_acc
                    best_val_UAR = Patient_UAR
                    best_val_F1 = Patient_f1
                    best_val_acc_soft = PCG_acc_soft_aver
                    best_val_soft = Soft_recall
                    best_epoch = epoch

                    best_Absent_recall = Absent_recall
                    best_Soft_recall = Soft_recall
                    best_Loud_recall = Loud_recall
                    best_Absent_f1 = Absent_f1
                    best_Soft_f1 = Soft_f1
                    best_Loud_f1 = Loud_f1
                    best_UAF = (best_Absent_f1 + best_Soft_f1 + best_Loud_f1) / 3
                    best_PCG_UAR = (Absent_recall + Soft_recall + Loud_recall) / 3
                    best_PCG_f1 = PCG_f1
                    best_Patient_f1 = Patient_f1

                    best_Absent_recall_patient = Absent_recall_patient
                    best_Soft_recall_patient = Soft_recall_patient
                    best_Loud_recall_patient = Loud_recall_patient
                    best_Absent_f1_patient = Absent_f1_patient
                    best_Soft_f1_patient = Soft_f1_patient
                    best_Loud_f1_patient = Loud_f1_patient
                    best_Patient_UAR = (Absent_recall_patient + Soft_recall_patient + Loud_recall_patient) / 3
                    best_Patient_WAR = Patient_WAR
                    result_path = os.path.join(model_result_path, "ResultFile")
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)

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

                    # 患者混淆矩阵

                    plt.figure()
                    plt.imshow(cm1, cmap=plt.cm.Blues)
                    plt.colorbar()
                    # 显示矩阵元素的数值
                    for i in range(cm1.shape[0]):
                        for j in range(cm1.shape[1]):
                            if i == 0 and j == 0:
                                plt.text(j, i, cm1[i, j], color='white', ha='center', va='center')
                            else:
                                plt.text(j, i, cm1[i, j], ha='center', va='center')
                    plt.xlabel('Predicted labels')
                    plt.ylabel('True labels')
                    plt.xticks([0, 1, 2], ['absent', 'soft', 'loud'])
                    plt.yticks([0, 1, 2], ['absent', 'soft', 'loud'])
                    plt.title('Confusion matrix')
                    plt.savefig(result_path + '/patient Confusion matrix.png', dpi=600)
                    plt.close()
                else:
                    no_better_epoch = no_better_epoch + 1

                # 保存验证集loss最小时的模型
                if loss_metric < best_val_loss:

                    torch.save(model, os.path.join(model_path, "loss_model"))
                    # sd：保存模型和参数
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(model_path, 'sd_loss_model.pth'))
                    print(
                        "Saving loss_model model to:",
                        os.path.join(model_path, "loss_model"),
                    )

                    min_loss_epoch = epoch
                    best_val_loss = loss_metric

                aver_PCG_acc.append(acc_metric)
                aver_PCG_UAR.append(PCG_UAR)
                aver_PCG_absent.append(Absent_recall)
                aver_PCG_soft.append(Soft_recall)
                aver_PCG_loud.append(Loud_recall)

            # if no_better_epoch == 15 :
            #     break

        torch.save(
            model,
            os.path.join(model_path, 'last_model'),
        )
        # sd：保存模型和参数
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(model_path, 'sd_last_model.pth'))
        print(
            "Saving last_model model to:",
            os.path.join(model_path, 'last_model'),
        )

        # === 显示训练集和验证集loss曲线 ===
        plt.figure()
        plt.plot(all_train_loss, linewidth=1, label='Training Loss')
        plt.plot(all_val_loss, linewidth=1, label='Validation Loss')
        plt.title('Training and Validation Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend()

        plt.savefig(result_path + '/Training and Validation Loss.png', dpi=600)
        plt.close()

        # === 显示训练集和验证集acc曲线 ===
        plt.figure()
        plt.plot(all_train_acc, linewidth=1, label='Training Acc')
        plt.plot(all_val_acc, linewidth=1, label='Validation Acc')
        plt.title('Training and Validation Acc', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend()

        plt.savefig(result_path + '/Training and Validation acc.png', dpi=600)
        plt.close()

        # 保存历史loss到txt文件
        np_train_acc = np.array(all_train_acc).reshape((len(all_train_acc), 1))  # reshape是为了能够跟别的信息组成矩阵一起存储
        np_train_loss = np.array(all_train_loss).reshape((len(all_train_loss), 1))
        np_val_acc = np.array(all_val_acc).reshape((len(all_val_acc), 1))  # reshape是为了能够跟别的信息组成矩阵一起存储
        np_val_loss = np.array(all_val_loss).reshape((len(all_val_loss), 1))
        np_out = np.concatenate([np_train_acc, np_val_acc, np_train_loss, np_val_loss], axis=1)

        f = result_path + "/save_result.txt"
        # if not os.path.isdir(f):
        #     os.makedirs(f)
        mytime = datetime.now()
        with open(f, "a") as file:
            file.write("===============================================================================" + "\n")
            file.write(str(mytime) + "\n")
            # file.write("# encoder layers = " + str(encoders) + "\n")
            # file.write("# img_size = " + str(img_size) + "\n")
            # file.write("# patch size = " + str(patch_size) + "\n")
            # file.write("# num_heads = " + str(num_heads) + "\n")
            file.write("# num_epochs = " + str(num_epochs) + "\n")
            # file.write("# train_batch_size = " + str(train_batch_size) + "\n")
            # file.write("# test_batch_size = " + str(test_batch_size) + "\n")
            file.write("# learning_rate = " + str(learning_rate) + "\n")
            file.write("# best_epoch = " + str(best_epoch) + "\n")
            file.write("# min_loss_epoch = " + str(min_loss_epoch) + "\n")
            file.write("# train_acc = " + str('{:.4f}'.format(best_train_acc)) + "\n")
            file.write("# val_acc = " + str('{:.4f}'.format(best_val_acc)) + "\n")
            file.write("# val_patient_acc = " + str('{:.4f}'.format(best_val_patient_acc)) + "\n")
            file.write("-----------------average_results----------------- " + "\n")
            file.write("Absent: " + str('{:.4f}'.format(sum(aver_PCG_absent) / len(aver_PCG_absent)))
                       + "  Soft: " + str('{:.4f}'.format(sum(aver_PCG_soft) / len(aver_PCG_soft)))
                       + "  Loud: " + str('{:.4f}'.format(sum(aver_PCG_loud) / len(aver_PCG_loud)))
                       + "  PCG_UAR: " + str('{:.4f}'.format(sum(aver_PCG_UAR) / len(aver_PCG_UAR)))
                       + "  PCG_Acc: " + str('{:.4f}'.format(sum(aver_PCG_acc) / len(aver_PCG_acc)))
                       + "\n")
            file.write("-----------------PCG_vali_recall----------------- " + "\n")
            file.write("Absent: " + str('{:.4f}'.format(best_Absent_recall))
                       + "  Soft: " + str('{:.4f}'.format(best_Soft_recall))
                       + "  Loud: " + str('{:.4f}'.format(best_Loud_recall))
                       + "  PCG_UAR: " + str('{:.4f}'.format(best_PCG_UAR))
                       + "\n")
            file.write("-------------------PCG_vali_F1------------------- " + "\n")
            file.write("Absent: " + str('{:.4f}'.format(best_Absent_f1))
                       + "  Soft: " + str('{:.4f}'.format(best_Soft_f1))
                       + "  Loud: " + str('{:.4f}'.format(best_Loud_f1))
                       + "  UAF: " + str('{:.4f}'.format(best_UAF))
                       + "\n")
            file.write("-----------------patient_vali_recall----------------- " + "\n")
            file.write("Absent: " + str('{:.4f}'.format(best_Absent_recall_patient))
                       + "  Soft: " + str('{:.4f}'.format(best_Soft_recall_patient))
                       + "  Loud: " + str('{:.4f}'.format(best_Loud_recall_patient))
                       + "  Patient_UAR: " + str('{:.4f}'.format(best_Patient_UAR))
                       + "  Patient_WAR: " + str('{:.4f}'.format(best_Patient_WAR))
                       + "\n")
            file.write("-------------------patient_vali_F1------------------- " + "\n")
            file.write("Absent: " + str('{:.4f}'.format(best_Absent_f1_patient))
                       + "  Soft: " + str('{:.4f}'.format(best_Soft_f1_patient))
                       + "  Loud: " + str('{:.4f}'.format(best_Loud_f1_patient))
                       + "  Average: " + str('{:.4f}'.format(best_Patient_f1))
                       + "\n")
            # file.write('train_acc    val_acc   train_loss    val_loss' + "\n")
            # for i in range(len(np_out)):
            #     file.write(str(np_out[i]) + '\n')
        print("save result successful!!!")
