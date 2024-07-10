import os
import numpy as np
# import scipy as sp, scipy.io, scipy.io.wavfile
from tqdm import tqdm

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


# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold()==str(y).strip().casefold()
    except AttributeError: # For Python 2.x compatibility
        return str(x).strip().lower()==str(y).strip().lower()


# Find patient data files.
# 将数据文件名后缀去掉后升序排列后加入列表filenames后返回
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension == '.txt':
            filename = os.path.join(data_folder, f)  # 获取datafolder中子文件完整路径
            filenames.append(filename),

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames


# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data


# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i == 0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id


# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations


# get murmur from patient data.输入为个体的.txt文件
def get_murmur(data):
    murmur = None
    for l in data.split('\n'):
        if l.startswith('#Murmur:'):
            try:
                murmur = l.split(': ')[1]
            except:
                pass
    if murmur is None:
        raise ValueError('No murmur available. Is your code trying to load labels from the hidden data?')
    return murmur


def get_grade(data):
    grade = None
    murmur = get_murmur(data)
    if not murmur == "Unknown":  # 排除Unknown数据
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


def get_murmur_locations(data):  # 返回带+的全部存在murmur的听诊区
    murmur_locations = None
    for text in data.split("\n"):
        if text.startswith("#Murmur locations:"):
            murmur_locations = text.split(": ")[1].strip()
    return murmur_locations


def get_label(data_directory):  # 按照ID顺序返回所有3s段label标签和听诊区位置、对应ID
    label = list()
    location = list()
    id = list()

    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' get label:'):
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            the_location = root.split("_")[1].strip()
            location.append(the_location)

            the_id = root.split("_")[0].strip()
            id.append(the_id)

            # 将不存在杂音的听诊位置标为absent
            the_label = 'Absent'
            txt_data = load_patient_data(os.path.join(data_directory, the_id + '.txt'))
            murmur_locations = (get_murmur_locations(txt_data)).split("+")
            for i in range(len(murmur_locations)):
                if the_location == murmur_locations[i]:
                    the_label = root.split("_")[2].strip()
            # the_label = root.split("_")[2].strip()
            if the_label == 'Absent':
                grade = 0
            elif the_label == 'Soft':
                grade = 1
            elif the_label == 'Loud':
                grade = 2
            label.append(grade)

    return np.array(label), np.array(location), np.array(id)


# 获取.wav文件index
def get_index(data_directory):
    index = list()
    i = 0
    for f in sorted(os.listdir(data_directory)):
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            index.append(i)
            i = i+1
        else:
            continue
    return np.array(index)



