from sklearn.metrics import classification_report, f1_score
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_closing
import numpy as np
from copy import copy
import sys
from GT_related import get_score_gt,extract_gt_label
import scipy.io


def eliminate_small_bout(pred,DE_size):
    erosion_array = binary_erosion(pred,iterations=DE_size).astype(pred.dtype)
    dilation_array = binary_dilation(erosion_array,iterations=DE_size).astype(pred.dtype)
    return dilation_array

def eliminate_gaps(pred,DE_size):
    dilation_array = binary_dilation(pred,iterations=DE_size).astype(pred.dtype)
    erosion_array = binary_erosion(dilation_array,iterations=DE_size).astype(pred.dtype)
    return erosion_array

if __name__ == "__main__":
    #best = 233
    behavior_name = 'stay_close'
    search_start = int(sys.argv[1])
    search_end = int(sys.argv[2])
    log_name = '%s_log_%s_%s.txt' % (behavior_name, search_start, search_end)
    #Learn = spio.loadmat('chase_Score_GT.mat', squeeze_me=True)
    #score_temp = Learn['score']
    #label_temp = Learn['label']
    label = extract_gt_label(behavior_name,'Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_%s_042718.txt' % behavior_name)
    label = label.loc[label['video_folder_path_abs'].str.contains('02_cam_')]
    result = get_score_gt(label)
    score_temp = result['score']
    label_temp = result['GT']
    unique_score = np.unique(score_temp)
    loop = np.linspace(min(unique_score),max(unique_score),1000)
    max_f1 = 0
    progress = 0
    #for cutoff in np.concatenate((unique_score[233:234],unique_score[search_start:search_end])):
    for cutoff in loop[search_start:search_end]:
        score = np.array(copy(score_temp))
        label = copy(label_temp)
        score[score_temp < cutoff] = 0
        score[score_temp >= cutoff] = 1
        continuous = np.diff(np.flatnonzero(np.concatenate(([True], score[1:]!= score[:-1], [True] ))))
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
                    dilation_record = dilation
                    erosion_record  = erosion
                    cutoff_record = cutoff
                    print(classification_report(label_to_evaluate, pred_to_evaluate), file=open(log_name, "a"))
                    print(classification_report(label_to_evaluate, pred_to_evaluate))
        progress += 1
        print(progress, file=open(log_name, "a"))
        print(progress)

    print('f1 = %s' % max_f1, file=open(log_name, "a"))
    print('dilation is %s ' % dilation_record, file=open(log_name, "a"))
    print('erosion is %s ' % erosion_record, file=open(log_name, "a"))
    print('cutoff = %s' % cutoff_record, file=open(log_name, "a"))
    print('f1 = %s' % max_f1)
    print('dilation is %s ' % dilation_record)
    print('erosion is %s ' % erosion_record)
    print('cutoff = %s' % cutoff_record)

    #bouts = [idx for idx in range(len(dilation_array)) if dilation_array[idx] == 1]

    #for k, g in groupby(enumerate(bouts), lambda ix : ix[0] - ix[1]):
    #    print(list(map(itemgetter(1), g)))

    #spio.savemat('B1_pred.mat', mdict={'Pred': y_pred})

    #print([idx for idx in range(len(Pred)) if Pred[idx] == 1])
