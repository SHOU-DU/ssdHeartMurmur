# ssdHeartMurmur
public heartmurmur
## 611commit
another try
    
# 大论文开发日志
## 2025-11-06 PM
最长测试集心音节拍点数：6401, 最短1156    
最长训练-验证集心音节拍点数：7009, 最短1142     
数据集划分时采用7200作为统一长度，原数据采样率为4kHz  
由于最长心音节拍与普通心音节拍相比长太多，需要填充的数据太多，故改为寻找适合的心音节拍长度作为训练数据

## 2026.1.16 PM
添加move_present.py文件，用于将present文件移动到指定文件夹
D:\sdmurmur\Qwen2Audio\murmur_data\test_ad_16k 存放不去除0的3s段重采样16k全部测试集数据
D:\sdmurmur\Qwen2Audio\murmur_data\train_vali_ad_16k 存放不去除0的3s段重采样16k全部训练，验证集数据
