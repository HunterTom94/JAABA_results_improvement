from GT_related import get_score_gt, extract_gt_label, get_score
import scipy.io as spio
from copy import copy
from sklearn.metrics import classification_report, f1_score
from New_parameters_determine import eliminate_gaps, eliminate_small_bout
import pandas as pd
import numpy as np
import os


def apply_para(cutoff, dilation, erosion, score_temp):
    score = np.array(copy(score_temp))
    score[score_temp < cutoff] = 0
    score[score_temp >= cutoff] = 1
    prediction = eliminate_gaps(eliminate_small_bout(score, erosion), dilation)
    prediction.astype(int)
    return prediction


def get_video_dir():
    root_path = 'Z:\April\JAABA\Experiments\CS'
    for (dirpath, dirnames, filenames) in os.walk(root_path):
        conditions = dirnames
        break
    conditions.remove('Orco')
    list = []
    for condition in conditions:
        for (dirpath, dirnames, filenames) in os.walk('%s\%s' % (root_path, condition)):
            videos = dirnames
            break
        for video in videos:
            list.append('%s\%s\%s' % (root_path, condition, video))
    return list


def PredictionGenerator(behavior_list):
    videos = get_video_dir()
    num_row = len(behavior_list) * len(videos)
    count = 0
    obj_arr = np.zeros((num_row, 4), dtype=np.object)
    for behavior in behavior_list:
        para = pd.read_excel('Parameters.xlsx', sheet_name=0)
        cutoff = para.loc[para['behavior'] == behavior]['cutoff']
        dilation = para.loc[para['behavior'] == behavior]['dilation']
        erosion = para.loc[para['behavior'] == behavior]['erosion']
        for video in videos:
            prediction = apply_para(float(cutoff), int(dilation), int(erosion), get_score(video, behavior))
            obj_arr[count, 0] = video.split('\\')[-1]
            obj_arr[count, 1] = video.split('\\')[-2]
            obj_arr[count, 2] = behavior
            obj_arr[count, 3] = prediction
            count += 1
    spio.savemat('Prediction082918.mat', mdict={'Prediction': obj_arr})
    return


def get_best_tuned(discon=None):
    behavior_names = ['chase', 'wing_search', 'wing_sing', 'touch', 'stay_close', 'still']
    best_dataset = pd.DataFrame()
    for name in behavior_names:
        if discon is None:
            tuned_csv = pd.read_csv('continuous trainer tuned result\%s_parameter_tune' % name, index_col=0)
        else:
            tuned_csv = pd.read_csv('discon_score_%s_parameter_tune' % name, index_col=0)
        to_be_concated = tuned_csv.loc[tuned_csv['f1'] == max(tuned_csv['f1'])].reset_index(drop=True)
        to_be_concated.insert(loc=0, column="behavior_name", value=[name] * to_be_concated.shape[0])
        best_dataset = pd.concat([best_dataset, to_be_concated])
    return best_dataset.loc[0]


def quick_f1(cutoff, dilation, erosion, score_temp, label_temp):
    score = np.array(copy(score_temp))
    label = copy(label_temp)
    score[score_temp < cutoff] = 0
    score[score_temp >= cutoff] = 1
    if dilation == 0 and erosion == 0:
        print('1')
        DE_output = score  # Test cutoff only
    else:
        print('2')
        DE_output = eliminate_gaps(eliminate_small_bout(score, erosion), dilation)
    # print(DE_output[17729:17839])
    pred_to_evaluate = DE_output[label != -1]
    label_to_evaluate = label[label != -1]

    print(classification_report(label_to_evaluate, pred_to_evaluate))


def getConfusionDetail(cutoff, dilation, erosion, score_gt, type):
    video_group = score_gt.groupby(["video_folder_path_abs"])
    all = pd.DataFrame()
    for video_name in score_gt["video_folder_path_abs"].unique():
        video_score_gt = video_group.get_group(video_name)
        video = []
        score_temp = video_score_gt['score']
        label_temp = video_score_gt['GT']
        score = copy(score_temp)
        label = copy(label_temp)
        score[score_temp < cutoff] = 0
        score[score_temp >= cutoff] = 1
        DE_output = eliminate_gaps(eliminate_small_bout(score, erosion), dilation)
        if type == 'FP':
            Type = np.array(np.logical_and(np.array(label - DE_output == -1, dtype=bool),
                                           np.array(label + DE_output == 1, dtype=bool), dtype=int))
        elif type == 'FN':
            Type = np.array(np.logical_and(np.array(label - DE_output == 1, dtype=bool),
                                           np.array(label + DE_output == 1, dtype=bool), dtype=int))
        elif type == 'TP':
            Type = np.array(np.logical_and(np.array(label - DE_output == 0, dtype=bool),
                                           np.array(label + DE_output == 2, dtype=bool), dtype=int))
        elif type == 'TN':
            Type = np.array(np.logical_and(np.array(label - DE_output == 0, dtype=bool),
                                           np.array(label + DE_output == 0, dtype=bool), dtype=int))
        continuous = np.diff(np.flatnonzero(np.concatenate(([True], Type[1:] != Type[:-1], [True]))))
        cumsum_continuous = np.append(np.array([1]), np.cumsum(continuous))
        if Type[0] == 0:
            type_length = continuous[1::2]
            type_odd = True
        elif Type[0] == 1:
            type_length = continuous[::2]
            type_odd = False
        for i in range(len(type_length)):
            keys = ('end', 'start', "video_folder_path_abs")
            if type_odd:
                values = (cumsum_continuous[2 * i + 2], cumsum_continuous[2 * i + 1], video_name)
            else:
                values = (cumsum_continuous[2 * i + 1], cumsum_continuous[2 * i], video_name)
            video.append(dict(zip(keys, values)))
        all = pd.concat([all, pd.DataFrame(video)])
    all['start'] = all['start'].apply(lambda x: x - 1)
    all.insert(loc=0, value=[type] * all.shape[0], column='Type')
    return all


def evaluate_parameters(number, behavior_name, test_vedio, no_cutoff=0, no_DE=0):
    # para_set = pd.read_csv('#%s_%s_parameter_tune' % (number,behavior_name),index_col=0)
    para_set = pd.read_csv('%s_parameter_tune' % (behavior_name), index_col=0)
    best_para = para_set.loc[para_set['f1'] == max(para_set['f1'])].reset_index(drop=True).loc[0]
    print(best_para)
    # GT = extract_gt_label(behavior_name,'Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_%s_042718.txt' % behavior_name)
    GT = extract_gt_label(behavior_name, 'Z:\#Yinan\Behavior Transition\GT\\4th_batch\GT_%s_082218.txt' % behavior_name)
    filtered_GT = pd.DataFrame()
    for vedio_name in test_vedio:
        filtered_GT = pd.concat([filtered_GT, GT.loc[(GT['video_folder_path_abs'].str.contains(vedio_name))]])
    filtered_GT = get_score_gt(filtered_GT)
    score_temp = filtered_GT['score']
    label_temp = filtered_GT['GT']
    if no_cutoff:
        print('1')
        quick_f1(0, 0, 0, score_temp, label_temp)
    elif no_DE:
        print('2')
        quick_f1(best_para['cutoff'], 0, 0, score_temp, label_temp)
    else:
        print('3')
        quick_f1(best_para['cutoff'], int(best_para['dilation']), int(best_para['erosion']), score_temp, label_temp)
        type_dict = {}
        for type in ['TP', 'TN', 'FP', 'FN']:
            temp = getConfusionDetail(best_para['cutoff'], int(best_para['dilation']), int(best_para['erosion']),
                                      filtered_GT, type)
            num = 0
            for index, row in temp.iterrows():
                num += row['end'] - row['start']
            type_dict[type] = num
        FNR = type_dict['FN'] / (type_dict['FN'] + type_dict['TP']) * 100
        print('FNR = %s' % FNR)
        FPR = type_dict['FP'] / (type_dict['FP'] + type_dict['TN']) * 100
        print('FPR = %s' % FPR)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    # PredictionGenerator(['chase', 'walk', 'sing', 'touch', 'stay_close', 'search', 'still', 'reorient','water'])
    PredictionGenerator(['search'])
    exit()
    number = '1'
    behavior_name = 'water'
    test_vedio = []
    # test_vedio.append('01_cam_20161019T103347')
    # test_vedio.append('01_cam_20161019T111429')
    # test_vedio.append('01_cam_20170721T142032')
    # test_vedio.append('02_cam_20161019T113144')
    # test_vedio.append('02_cam_20161019T115202')
    # test_vedio.append('02_cam_20161019T114234')
    # test_vedio.append('10_cam_20161103T111653')
    # test_vedio.append('19_cam_20170721T144256')
    # test_vedio.append('09_cam_20161110T100639')
    # test_vedio.append('10_cam_20161110T103357')
    # test_vedio.append('12_cam_20161117T101918')

    test_vedio.append('04_cam_20161026T111729')
    test_vedio.append('18_cam_20170721T140126')
    test_vedio.append('19_cam_20170728T140339')
    test_vedio.append('09_cam_20161103T104443')
    test_vedio.append('11_cam_20161110T105008')
    test_vedio.append('15_cam_20170413T170304')

    evaluate_parameters(number, behavior_name, test_vedio)
    exit()

    # quick_f1(best_result['cutoff'],int(best_result['dilation']),int(best_result['erosion']),score_temp,label_temp)
    # getConfusionDetail(1.77484809,1,4,result,'FN')

    # print(result.loc[result['f1'] == 0.6844518549395582])

    # behavior_names = ['chase','wing_search','wing_sing','touch','stay_close','still']
    # best_dataset_con = pd.DataFrame()
    # best_dataset_discon = pd.DataFrame()
    # for name in behavior_names:
    #     result = pd.read_csv('discon_score_%s_parameter_tune' % name,index_col=0)
    #     best_dataset_discon = pd.concat([best_dataset_discon, get_best_tuned(result)])
    #     result = pd.read_csv('continuous trainer tuned result\%s_parameter_tune' % name,index_col=0)
    #     best_dataset_con = pd.concat([best_dataset_con, get_best_tuned(result)])
    # print(get_best_tuned())
    # print(best_dataset_discon.loc[0])
