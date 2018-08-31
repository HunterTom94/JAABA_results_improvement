from GT_related import extract_gt_label,get_score_gt
from PerformanceTest import getConfusionDetail
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PerformanceTest import get_best_tuned

def draw_gt(gt_df):
    y_cursor = 0
    video_group = gt_df.groupby(["video_folder_path_abs"])
    for video_name in gt_df["video_folder_path_abs"].unique():
        video_label = video_group.get_group(video_name)
        for index, row in video_label.iterrows():
            start = row["bout_start_frame_index"]
            end = row["bout_end_frame_index"]
            length = end - start
            # if row["batch#"] == '2nd':
            #     axarr[0].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'none', edgecolor= 'r', linewidth=0.2))
            # elif row["batch#"] == '3rd':
            #     if row['bout_or_not'] != 'None':
            #         axarr[0].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'b'))
            #     else:
            #         axarr[0].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'r'))
            if row['bout_or_not'] != 'None':
                    axarr[0].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'b'))
            else:
                axarr[0].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'r'))
        y_cursor = y_cursor + 1

def draw_true():
    return

def drawConfusionType(df):
    y_cursor = 0
    video_group = df.groupby(["video_folder_path_abs"])
    for video_name in df["video_folder_path_abs"].unique():
        video_label = video_group.get_group(video_name)
        for index, row in video_label.iterrows():
            start = row["start"]
            end = row["end"]
            length = end - start
            if row["Type"] == 'FN':
                axarr[1].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'r'))
            elif row["Type"] == 'FP':
                axarr[1].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'b'))
            elif row["Type"] == 'TN':
                axarr[0].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'r'))
            elif row["Type"] == 'TP':
                axarr[0].add_patch(patches.Rectangle((start, y_cursor),length,1,fc = 'b'))
        y_cursor = y_cursor + 1
    return [max(df['end']),y_cursor]

def drawConfusionTypeExe(behavior_name):
    f.suptitle('%s_discon' % behavior_name)
    axarr[0].title.set_text('TP & TN')
    axarr[1].title.set_text('FP & FN')
    axarr[0].set_xlabel('frame_index')
    axarr[1].set_xlabel('frame_index')
    axarr[0].set_ylabel('video_index')
    axarr[1].set_ylabel('video_index')
    x_lim = 0
    y_lim = 0
    label = extract_gt_label(behavior_name,'Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_%s_042718.txt' % behavior_name)
    label = label.loc[label['video_folder_path_abs'].str.contains('02_cam_')]
    score_gt = get_score_gt(label)
    tuned_results = get_best_tuned(1)
    cutoff= float(tuned_results.loc[tuned_results['behavior_name'] == behavior_name]['cutoff'])
    dilation=int(tuned_results.loc[tuned_results['behavior_name'] == behavior_name]['dilation'])
    erosion=int(tuned_results.loc[tuned_results['behavior_name'] == behavior_name]['erosion'])
    [x,y] = drawConfusionType(getConfusionDetail(cutoff, dilation,erosion,score_gt,'FP'))
    x_lim = max(x,x_lim)
    y_lim = max(y,y_lim)
    [x,y] = drawConfusionType(getConfusionDetail(cutoff, dilation,erosion,score_gt,'FN'))
    x_lim = max(x,x_lim)
    y_lim = max(y,y_lim)
    [x,y] = drawConfusionType(getConfusionDetail(cutoff, dilation,erosion,score_gt,'TN'))
    x_lim = max(x,x_lim)
    y_lim = max(y,y_lim)
    [x,y] = drawConfusionType(getConfusionDetail(cutoff, dilation,erosion,score_gt,'TP'))
    x_lim = max(x,x_lim)
    y_lim = max(y,y_lim)
    axarr[0].set_xlim([0,x_lim])
    axarr[0].set_ylim([0,y_lim])
    return [x_lim,y_lim]

if __name__ == "__main__":
    f, axarr = plt.subplots(2, sharex=True,sharey=True)
    #second_gt = extract_gt_label('chase','Z:\#Yinan\Behavior Transition\GT\\2nd_batch\chase_021218_GT.txt')
    # third_gt = extract_gt_label('chase','Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_stay_close_042718.txt')
    #second_gt.insert(loc=0,column = "batch#",  value = ["2nd"]*second_gt.shape[0])
    # third_gt.insert(loc=0,column = "batch#",  value = ["3rd"]*third_gt.shape[0])
    #all_gt = pd.concat([second_gt,third_gt])
    #draw_gt(all_gt)
    behavior_name = 'sing'
    GT = extract_gt_label(behavior_name,'Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_%s_042718.txt' % behavior_name)
    draw_gt(GT)
    axarr[0].set_xlim([0,max(GT["bout_end_frame_index"])])
    axarr[0].set_ylim([0,len(GT["video_folder_path_abs"].unique())])




    # print(drawConfusionTypeExe(behavior_name))
    # f.savefig('test.png')



    plt.show()

