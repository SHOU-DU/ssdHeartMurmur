import os
import torch
import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
import torch.nn as nn
def cal_patient_acc(path,all_id,all_y_pred,all_location):
    # soft = nn.Softmax(dim=0)
    # all_y_pred = soft(all_y_pred)
    patient_files = find_patient_files(path)
    num_patient_files = len(patient_files)
    ture_label = list()  # 存放病人的真实标签
    pre_label = list()  # 存放病人的预测标签
    num_correct_patient = 0
    for i in range(num_patient_files):
        current_patient_data = load_patient_data(patient_files[i])
        # num_location = get_num_locations(current_patient_data)
        current_ID = current_patient_data.split(" ")[0]
        current_locations = list(set(get_locations(current_patient_data)))
        num_location = len(current_locations)
        current_patient_pre = torch.zeros(num_location, 3)
        current_patient_label = get_grade_number(current_patient_data)
        # 分别计算当前ID下听诊区各个片段的预测概率的和
        for n in range(len(all_id)):
            if all_id[n] == current_ID:
                if all_location[n] in current_locations:
                    j = current_locations.index(all_location[n])
                    current_patient_pre[j] += all_y_pred[n]
        _, current_patient_pre_label = current_patient_pre.max(1)
        # 计算该患者最终的label预测结果
        current_label = 0
        num_zeros = 0
        for n in range(len(current_patient_pre_label)):
            if current_patient_pre_label[n] == 0:
                num_zeros += 1
        # if  num_location >= 4:  # 所听诊区都无杂音
        #     current_label = max(current_patient_pre_label)
        # elif num_zeros == num_location:  # 所有听诊区都有杂
        #     current_label = 0
        # else:
        #     current_label = 1
        if num_zeros == num_location:  # 所听诊区都无杂音
            current_label = 0
        elif num_zeros == 0:  # 所有听诊区都有杂音
            current_label = max(current_patient_pre_label)
        else:
            current_label = 1
        # current_label = max(current_patient_pre_label)
        if current_label == current_patient_label:
            num_correct_patient += 1
        pre_label.append(current_label)
        ture_label.append(current_patient_label)

    patient_acc = num_correct_patient / num_patient_files
    # print("patient Acc: %.4f" % patient_acc)
    return pre_label, ture_label,patient_acc

#由单个听诊区结果计算患者水平的结果
def single_result(path,all_id,all_y_pred):
    ture_label = list()  # 存放病人的真是标签
    pre_label = list()  # 存放病人的预测标签
    num_correct_patient = 0
    patient_id = set(all_id)#获取所有id集合
    for current_ID in patient_id: #整合片段结果
        current_patient_pre = torch.zeros(1, 3)
        txtFile = os.path.join(path,current_ID+".txt")
        current_patient_data = load_patient_data(txtFile)
        current_patient_label = get_grade_number(current_patient_data)#获得当前患者真实标签
        for n in range(len(all_id)):
            if all_id[n] == current_ID:
                current_patient_pre += all_y_pred[n]
        _, current_patient_pre_label = current_patient_pre.max(1)
        current_label = 0
        if current_patient_pre_label == current_patient_label:
            num_correct_patient += 1
        pre_label.append(torch.squeeze(current_patient_pre_label))
        ture_label.append(current_patient_label)
    patient_acc = num_correct_patient / len(patient_id)
    # print("patient Acc: %.4f" % patient_acc)
    return pre_label, ture_label, patient_acc

def location_result(all_id,all_y_pred,all_location,all_label):
    AV_id = list()
    MV_id = list()
    TV_id = list()
    PV_id = list()

    AV_pred = list()
    MV_pred = list()
    TV_pred = list()
    PV_pred = list()

    AV_label = list()
    MV_label = list()
    TV_label = list()
    PV_label = list()
    for i in range(len(all_id)):
        if(all_location[i] == 'AV'):
            AV_id.append(all_id[i])
            AV_pred.append(all_y_pred[i])
            AV_label.append(all_label[i])
        if (all_location[i] == 'MV'):
            MV_id.append(all_id[i])
            MV_pred.append(all_y_pred[i])
            MV_label.append(all_label[i])
        if (all_location[i] == 'TV'):
            TV_id.append(all_id[i])
            TV_pred.append(all_y_pred[i])
            TV_label.append(all_label[i])
        if (all_location[i] == 'PV'):
            PV_id.append(all_id[i])
            PV_pred.append(all_y_pred[i])
            PV_label.append(all_label[i])
    AV_ture_label, AV_pre_label = cal_patient_single(AV_id,AV_label,AV_pred)
    MV_ture_label, MV_pre_label = cal_patient_single(MV_id, MV_label, MV_pred)
    TV_ture_label, TV_pre_label = cal_patient_single(TV_id, TV_label, TV_pred)
    PV_ture_label, PV_pre_label = cal_patient_single(PV_id, PV_label, PV_pred)

    return (AV_ture_label, AV_pre_label,
            MV_ture_label, MV_pre_label,
            TV_ture_label, TV_pre_label,
            PV_ture_label, PV_pre_label)
#由单个听诊区计算患者维度结果
def cal_patient_single(all_id,all_label,all_pred):
    ture_label = list()  # 存放病人的真是标签
    pre_label = list()
    current_id = all_id[0]
    current_label = all_label[0]
    current_pred = np.zeros(3)
    for i in range(len(all_id)):
        if all_id[i] == current_id:
            current_pred += all_pred[i]
        else:
            current_pre_label = np.argmax(current_pred)
            pre_label.append(current_pre_label)
            ture_label.append(current_label)
            current_id = all_id[i]
            current_label = all_label[i]
            current_pred = np.zeros(3)

    pre_label.append(current_pre_label)
    ture_label.append(current_label)
    return ture_label,pre_label


#返回该路径下的所有病人txt文件
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith(".") and extension == ".txt":
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(
            filenames, key=lambda filename: int(os.path.split(filename)[1][:-4])
        )

    return filenames

# Load patient data as a string.
#获取text文件文本内容
def load_patient_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    return data

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False


# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold() == str(y).strip().casefold()
    except AttributeError:  # For Python 2.x compatibility
        return str(x).strip().lower() == str(y).strip().lower()


# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = scipy.io.wavfile.read(filename)
    return recording, frequency


# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1 : num_locations + 1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(" ")
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings


# Get number of recording locations from patient data.
#听诊区数目
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split("\n")):
        if i == 0:
            num_locations = int(l.split(" ")[1])
        else:
            break
    return num_locations




# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, text in enumerate(data.split("\n")):
        entries = text.split(" ")
        if i == 0:
            pass
        elif 1 <= i <= num_locations:
            locations.append(entries[0])
        else:
            break
    return locations


# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = (
        str(x).replace('"', "").replace("'", "").strip()
    )  # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x) == 1) or (x in ("True", "true", "T", "t")):
        return 1
    else:
        return 0

# Get age from patient data.
def get_age(data):
    age = None
    for text in data.split("\n"):
        if text.startswith("#Age:"):
            age = text.split(": ")[1].strip()
    return age

def get_Murmur_locations(data):
    Murmur_locations = None
    for text in data.split("\n"):
        if text.startswith("#Murmur locations:"):
            Murmur_locations = text.split(": ")[1].strip()
    return Murmur_locations





# Get sex from patient data.
def get_sex(data):
    sex = None
    for text in data.split("\n"):
        if text.startswith("#Sex:"):
            sex = text.split(": ")[1].strip()
    return sex


# Get height from patient data.
def get_height(data):
    height = None
    for text in data.split("\n"):
        if text.startswith("#Height:"):
            height = float(text.split(": ")[1].strip())
    return height


# Get weight from patient data.
def get_weight(data):
    weight = None
    for text in data.split("\n"):
        if text.startswith("#Weight:"):
            weight = float(text.split(": ")[1].strip())
    return weight

def get_grade(data):
    grade = None
    for text in data.split("\n"):
        if text.startswith("#Systolic murmur grading:"):
            grading = text.split(": ")[1].strip()
            if grading == 'I/VI' or grading == 'II/VI':
                grade = 'Soft'
            elif grading == 'III/VI':
                grade = 'Loud'
            else:
                grade = 'Absent'
    return grade

def get_grade_number(data):
    grade = None
    for text in data.split("\n"):
        if text.startswith("#Systolic murmur grading:"):
            grading = text.split(": ")[1].strip()
            if grading == 'I/VI' or grading == 'II/VI':
                grade = 1
            elif grading == 'III/VI':
                grade = 2
            else:
                grade = 0
    return grade


# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for text in data.split("\n"):
        if text.startswith("#Pregnancy status:"):
            is_pregnant = bool(sanitize_binary_value(text.split(": ")[1].strip()))
    return is_pregnant

# Get murmur from patient data.
def get_murmur(data):
    murmur = None
    for text in data.split("\n"):
        if text.startswith("#Murmur:"):
            murmur = text.split(": ")[1]
    if murmur is None:
        raise ValueError(
            "No murmur available. Is your code trying to load labels from the hidden data?"
        )
    return murmur


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for text in data.split("\n"):
        if text.startswith("#Outcome:"):
            outcome = text.split(": ")[1]
    if outcome is None:
        raise ValueError(
            "No outcome available. Is your code trying to load labels from the hidden data?"
        )
    return outcome

# Extract features from the data.
def get_metadata(data):

    # Extract the age group and replace with the (approximate) number of months
    # for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, "Neonate"):
        age = 0.5
    elif compare_strings(age_group, "Infant"):
        age = 6
    elif compare_strings(age_group, "Child"):
        age = 6 * 12
    elif compare_strings(age_group, "Adolescent"):
        age = 15 * 12
    elif compare_strings(age_group, "Young Adult"):
        age = 20 * 12
    else:
        age = float("nan")

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, "Female"):
        sex_features[0] = 1
    elif compare_strings(sex, "Male"):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant]))

    return np.asarray(features, dtype=np.float32)
#去尖峰
def schmidt_spike_removal(original_signal, fs = 4000):
    windowsize = int(np.round(fs/4))
    trailingsamples = len(original_signal) % windowsize
    sampleframes = np.reshape(original_signal[0 : len(original_signal)-trailingsamples], (-1, windowsize) )
    MAAs = np.max(np.abs(sampleframes), axis = 1)
    while len(np.where(MAAs > np.median(MAAs)*3 )[0]) != 0:
        window_num = np.argmax(MAAs)
        spike_position = np.argmax(np.abs(sampleframes[window_num,:]))
        zero_crossing = np.abs(np.diff(np.sign(sampleframes[window_num, :])))
        if len(zero_crossing) == 0:
            zero_crossing = [0]
        zero_crossing = np.append(zero_crossing, 0)
        if len(np.nonzero(zero_crossing[:spike_position+1])[0]) > 0:
            spike_start = np.nonzero(zero_crossing[:spike_position+1])[0][-1]
        else:
            spike_start = 0
        zero_crossing[0:spike_position+1] = 0
        spike_end = np.nonzero(zero_crossing)[0][0]
        sampleframes[window_num, spike_start : spike_end] = 0.0001;
        MAAs = np.max(np.abs(sampleframes), axis = 1)
    despiked_signal = sampleframes.flatten()
    despiked_signal = np.concatenate([despiked_signal, original_signal[len(despiked_signal) + 1:]])
    return despiked_signal


def decision(current_patient_pre_label):
    num_zeros = 0
    num_location =len(current_patient_pre_label)
    for n in range(len(current_patient_pre_label)):
        if current_patient_pre_label[n] == 0:
            num_zeros += 1
    if num_zeros == num_location:  # 所听诊区
        current_label = 0
    elif num_location >= 4 and num_zeros == 0:  # 所有听诊区都有杂
        current_label = max(current_patient_pre_label)
    else:
        current_label = 1
    return current_label
