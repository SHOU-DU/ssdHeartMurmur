import librosa
import numpy as np

if __name__ == '__main__':
    csv_file = r'E:\sdmurmur\EnvelopeandSE60Hz\data_kfold_cut_zero\0_fold\train_data\2530_AV_Absent_0.csv'
    csv_data = np.loadtxt(csv_file, delimiter=',')
    csv_data_cut = csv_data[:, :-1]
    print(csv_data_cut.shape)