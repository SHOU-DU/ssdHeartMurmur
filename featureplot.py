# 画特征图比较各个特征图
import numpy as np

from kfold_feature_extraction import *
import matplotlib.pyplot as plt
import os
from spafe.utils.converters import erb2hz


if __name__ == '__main__':
    fig_path = 'feature_fig'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    kfold_festure_in = "data_kfold_out"  # 分好折的3s数据
    kfold_folder_vali = os.path.join(kfold_festure_in, "0_fold" + r"\vali_data")  # 五折交叉验证，最后通过测试集测试

    # 选择特征
    # vali_feature = Log_GF(kfold_folder_vali)
    # vali_feature = GF(kfold_folder_vali)
    vali_feature = Aweight_Log_GF(kfold_folder_vali)
    # vali_feature = log10_GF(kfold_folder_vali)
    # np.savetxt(r'gfreqs.txt', gfreqs, newline='\n')
    # 选择样本
    sample_100 = vali_feature[99]  # 第100个片段的特征图
    # sample_100 = vali_feature[100]  # 第101个片段的特征图
    plt.imshow(sample_100)
    plt.title('100_Aweight_Log_GF')
    plt.savefig(fig_path + r'\100_Aweight_Log_GF.png')
    plt.show()
