from sklearn.metrics import classification_report, f1_score
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import numpy as np
from copy import copy
import sys
from GT_related import get_score_gt,extract_gt_label
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from time import time



def eliminate_small_bout(pred,DE_size):
    erosion_array = binary_erosion(pred,iterations=DE_size).astype(pred.dtype)
    dilation_array = binary_dilation(erosion_array,iterations=DE_size).astype(pred.dtype)
    return dilation_array

def eliminate_gaps(pred,DE_size):
    dilation_array = binary_dilation(pred,iterations=DE_size).astype(pred.dtype)
    erosion_array = binary_erosion(dilation_array,iterations=DE_size).astype(pred.dtype)
    return erosion_array

def parallel_starter(behavior_name,test_vedio):
    num_cores = 15#multiprocessing.cpu_count()
    GT = extract_gt_label(behavior_name,'Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_%s_042718.txt' % behavior_name)
    GT = extract_gt_label(behavior_name, 'Z:\#Yinan\Behavior Transition\GT\\4th_batch\GT_%s_061418.txt' % behavior_name)
    print(np.unique(GT['video_folder_path_abs']))
    exit()
    filtered_GT = pd.DataFrame()
    for vedio_name in test_vedio:
        filtered_GT = pd.concat([filtered_GT, GT.loc[(GT['video_folder_path_abs'].str.contains(vedio_name))]])
    filtered_GT = get_score_gt(filtered_GT)
    score_temp = filtered_GT['score']
    label_temp = filtered_GT['GT']
    unique_score = np.unique(score_temp)
    loop = np.linspace(min(unique_score),max(unique_score),1000)
    parameter_set = Parallel(n_jobs=num_cores)(delayed(for_loop)(cutoff,score_temp,label_temp)for cutoff in loop[150:850])
    pd.DataFrame(parameter_set).to_csv('%s_parameter_tune' % behavior_name)
    print('finished')
    return

def for_loop(cutoff,score_temp,label_temp):
    start_time = time()
    max_f1 = 0
    score = np.array(copy(score_temp))
    label = copy(label_temp)
    score[score_temp < cutoff] = 0
    score[score_temp >= cutoff] = 1
    continuous = np.diff(np.flatnonzero(np.concatenate(([True], score[1:]!= score[:-1], [True] ))))
    if len(continuous) == 1:
        return
    if score[0] == 0:
        gap_length = continuous[::2]
        bout_length = continuous[1::2]
    elif score[0] == 1:
        bout_length = continuous[::2]
        gap_length = continuous[1::2]
    gap_length = np.linspace(min(gap_length),max(gap_length),100,dtype=int)
    bout_length = np.linspace(min(bout_length),max(bout_length),100,dtype=int)
    for erosion in bout_length:
        for dilation in gap_length:
            DE_output = eliminate_gaps(eliminate_small_bout(score,erosion),dilation)
            pred_to_evaluate = DE_output[label != -1]
            label_to_evaluate = label[label != -1]
            f1 = f1_score(label_to_evaluate, pred_to_evaluate)
            if f1 > max_f1:
                max_f1 = f1
                parameter_set = {'f1':f1,'dilation':dilation,'erosion':erosion,'cutoff':cutoff}
    print(time()-start_time)
    try:
        return parameter_set
    except UnboundLocalError:
        return
if __name__ == "__main__":
    behavior_name = 'walk'
    test_vedio = []
    test_vedio.append('01_cam_20161019T103347')
    test_vedio.append('01_cam_20161019T111429')
    # test_vedio.append('01_cam_20170721T142032')
    # test_vedio.append('02_cam_20161019T113144')
    test_vedio.append('02_cam_20161019T115202')
    test_vedio.append('02_cam_20161019T114234')

    parallel_starter(behavior_name,test_vedio)
