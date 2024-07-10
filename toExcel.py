import os

import numpy as np
from tqdm import tqdm
from patient_information import *
import pandas as pd

# Get weight from patient data.
def get_murmur_grade(data):
    grading = None
    for text in data.split("\n"):
        if text.startswith("#Systolic murmur grading:"):
            grading = text.split(": ")[1].strip()
    return grading


def get_height(data):
    height = None
    for text in data.split("\n"):
        if text.startswith("#Height:"):
            height = text.split(": ")[1].strip()
    return height


# Get weight from patient data.
def get_weight(data):
    weight = None
    for text in data.split("\n"):
        if text.startswith("#Weight:"):
            weight = text.split(": ")[1].strip()
    return weight

def get_Additional (data):
    Additional = None
    for text in data.split("\n"):
        if text.startswith("#Additional ID:"):
            Additional = text.split(": ")[1].strip()
    return Additional

def get_locations (data):
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

    return "+".join(locations)

data_directory = 'F:/heart_data/2022_challenge_new/the-circor-digiscope-phonocardiogram-dataset-1.0.3/test_data'
patient_files = find_patient_files(data_directory)
num_patient_files = len(patient_files)
patient_ID = list()
murmur = list()
outcome = list()
age = list()
sex = list()
height = list()
weight = list()
pregnant = list()
additional = list()
locations = list()


for i in tqdm(range(num_patient_files)):
    current_patient_data = load_patient_data(patient_files[i])
    murmur_unkonwn = get_murmur(current_patient_data)
    # 跳过unknown的数据
    if murmur_unkonwn == 'Unknown':
        continue
    current_ID = current_patient_data.split(" ")[0]  # 获取ID
    patient_ID.append(current_ID)  # 添加ID

    current_murmur = get_murmur_grade(current_patient_data)
    current_outcome = get_outcome(current_patient_data)
    current_age = get_age(current_patient_data)
    current_sex = get_sex(current_patient_data)
    current_height = get_height(current_patient_data)
    current_weight = get_weight(current_patient_data)
    current_pregnet = get_pregnancy_status(current_patient_data)
    current_additional = get_Additional(current_patient_data)
    current_locations = get_locations(current_patient_data)

    murmur.append(current_murmur)
    outcome.append(current_outcome)
    age.append(current_age)
    sex.append(current_sex)
    height.append(current_height)
    weight.append(current_weight)
    pregnant.append(current_pregnet)
    additional.append(current_additional)
    locations.append(current_locations)

patient_ID = np.vstack(patient_ID)
murmur = np.vstack(murmur)
outcome = np.vstack(outcome)
age = np.vstack(age)
sex = np.vstack(sex)
height = np.vstack(height)
weight = np.vstack(weight)
pregnant = np.vstack(pregnant)
additional = np.vstack(additional)
locations = np.vstack(locations)

patients_pd = pd.DataFrame(patient_ID, columns=['ID'])
murmur_pd = pd.DataFrame(murmur, columns=['murmur'])
outcome_pd = pd.DataFrame(outcome, columns=['outcome'])

age_pd = pd.DataFrame(age, columns=['age'])
sex_pd = pd.DataFrame(sex, columns=['sex'])
height_pd = pd.DataFrame(height, columns=['height'])
weight_pd = pd.DataFrame(weight, columns=['weight'])
pregnant_pd = pd.DataFrame(pregnant, columns=['pregnant'])
additional_pd = pd.DataFrame(additional,columns=["additonal"])
Locations_pd = pd.DataFrame(locations,columns=["Recording locations"])

complete_pd = pd.concat([patients_pd, murmur_pd,outcome_pd,age_pd, sex_pd,height_pd,weight_pd,pregnant_pd,additional_pd,Locations_pd], axis=1)
print(complete_pd)
complete_pd.to_excel('test_data.xlsx', index=False)



