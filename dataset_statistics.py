from helper_code import *


if __name__ == '__main__':

    original_dataset_folder = r"E:\sdmurmur\thecircordataset\test_data"
    patient_files = find_patient_files(original_dataset_folder)  # 获取升序排列后.txt文件路径列表
    num_patient_files = len(patient_files)
    # 检查是否读取到了原始数据
    if num_patient_files == 0:
        raise Exception('No data was provided.')
    classes = ['Present', 'Unknown', 'Absent']  # 杂音类别
    pAbsentID = []
    pPresentID = []
    pUnknowID = []
    pIDs = []
    plabel = []  # 个体标签
    for i in range(num_patient_files):
        current_patient_data = load_patient_data(patient_files[i])  # 加载对应个体.txt文件
        label = get_murmur(current_patient_data)  # 'Present', 'Unknown', 'Absent'
        pID = get_patient_id(current_patient_data)  # 个体ID
        # pIDs.append(pID)
        # plabel.append(classes.index(label))  # 按照ID读取顺序存储标签
        if label == classes[0]:
            pPresentID.append(pID)
            pIDs.append(pID)
            plabel.append(classes.index(label))  # 按照ID读取顺序存储标签
        elif label == classes[1]:
            pIDs.append(pID)
            pUnknowID.append(pID)
            # print('Unknown ID is:', pID)
        else:
            pAbsentID.append(pID)
            pIDs.append(pID)
            plabel.append(classes.index(label))  # 按照ID读取顺序存储标签
            # print('Absent ID is:', pID)
    print('Total patientID num:', len(pIDs))
    print('PresentID num:', len(pPresentID))
    print('AbsentID num:', len(pAbsentID))
    print('UnknownID num:', len(pUnknowID))
