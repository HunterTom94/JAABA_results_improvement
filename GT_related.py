import numpy as np
import scipy.io as spio
import pandas as pd
import math

def extract_gt_label(behavior_name,gt_path):
    label_df = pd.read_csv(gt_path,sep =" ", header=None, names=["video_folder_path_abs","fly_index",
                                                          "bout_start_frame_index","bout_end_frame_index","bout_or_not"])

    label_df.insert(loc = 0, column = "behavior_name",  value = [behavior_name]*label_df.shape[0])
    label_df["fly_index"] = label_df["fly_index"].apply(lambda x: x - 1)
    label_df["bout_start_frame_index"] = label_df["bout_start_frame_index"].apply(lambda x: x - 1)
    #label_df = label_df.drop([0])
    return label_df

def get_score(video,behavior):
    df = pd.read_excel('Z:\#Yinan\Behavior Transition\FlyLabels.xlsx', sheet_name=video.split('\\')[-2], header=None)
    row = df.iloc[int(video.split('\\')[-1][0:2]) - 1]
    b = (df.ix[row.name] == 1)
    index = b.index[b.argmax()]
    date = '080218'
    if behavior in ['water']:
        date = '082218'
    # elif behavior in ['stay_close']:
    #     date = '060218'
    # elif behavior in ['search']:
    #     date = '071218'
    # elif behavior in ['walk','still']:
    #     date = '072418'
    # elif behavior == 'chase':
    #     date = '061418_2'
    # elif behavior == 'reorient':
    #     date = '073018'

    score_len = min(len(spio.loadmat('%s\scores_%s_%s.mat' % (video, behavior,date), squeeze_me=False)['allScores']['scores'][0][0][0][0][0]),len(spio.loadmat('%s\scores_%s_%s.mat' % (video, behavior,date), squeeze_me=False)['allScores']['scores'][0][0][0][1][0]))
    score = spio.loadmat('%s\scores_%s_%s.mat' % (video, behavior,date), squeeze_me=False)['allScores']['scores'][0][0][0][index][0]
    print(video)
    print(score_len)

    return score[:score_len]

def get_score_gt(label_df):
    label_df = label_df.reset_index(drop=True)
    video_group = label_df.groupby(["video_folder_path_abs"])
    all_data = pd.DataFrame()
    for video_name in label_df["video_folder_path_abs"].unique():
        video_label = video_group.get_group(video_name)
        fly_index = label_df.loc[label_df['video_folder_path_abs'] == video_name].iloc[0]["fly_index"]
        # score = spio.loadmat('%s\scores_%s_070218.mat' % (video_name,label_df['behavior_name'][0]), squeeze_me=False)['allScores']['scores'][0][0][0][fly_index][0]
        behavior = label_df['behavior_name'][0]
        date = '082218'
        # if behavior in ['sing','touch']:
        #     date = '061418'
        # elif behavior in ['stay_close']:
        #     date = '060218'
        # elif behavior in ['search']:
        #     date = '071218'
        # elif behavior in ['walk','still']:
        #     date = '072418'
        # elif behavior == 'chase':
        #     date = '061418_2'
        # elif behavior == 'reorient':
        #     date = '073018'

        score = spio.loadmat('%s\scores_%s_%s.mat' % (video_name, behavior,date), squeeze_me=False)['allScores']['scores'][0][0][0][fly_index][0]
        score_gt = pd.DataFrame()
        score_gt.insert(loc = 0,value = score,column='score')
        score_gt.insert(loc = 0,value = [video_name]*score_gt.shape[0],column='video_folder_path_abs')
        gt = np.array([-1]* score_gt.shape[0])
        bout_or_not_group = video_label.groupby(["bout_or_not"])
        for if_bout in video_label["bout_or_not"].unique():
            bout_or_not = bout_or_not_group.get_group(if_bout)
            for index, row in bout_or_not.iterrows():
                start = row["bout_start_frame_index"]
                end = row["bout_end_frame_index"]
                if if_bout == "None":
                    gt[start:end] = 0
                else:
                    gt[start:end] = 1
        score_gt.insert(loc = 2,value = gt,column='GT')
        all_data = pd.concat([all_data,score_gt])
    return all_data

def get_gt_matlab(behavior_list):
    num_row = 0
    for behavior in behavior_list:
        if behavior in ['chase','sing','touch','walk']:
            GT = extract_gt_label(behavior,'Z:\#Yinan\Behavior Transition\GT\\4th_batch\GT_%s_061418.txt' % behavior)
        else:
            GT = extract_gt_label(behavior,'Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_%s_042718.txt' % behavior)
        num_row += len(np.unique(GT['video_folder_path_abs']))

    count = 0
    obj_arr = np.zeros((num_row, 3), dtype=np.object)
    for behavior in behavior_list:
        if behavior in ['chase','sing','touch','walk']:
            GT = extract_gt_label(behavior,'Z:\#Yinan\Behavior Transition\GT\\4th_batch\GT_%s_061418.txt' % behavior)
        else:
            GT = extract_gt_label(behavior,'Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_%s_042718.txt' % behavior)
        GT = get_score_gt(GT)
        for video in np.unique(GT['video_folder_path_abs']):
            video_db = GT.loc[GT['video_folder_path_abs'] == video]
            obj_arr[count, 0] = video
            obj_arr[count, 1] = behavior
            obj_arr[count, 2] = video_db['GT'].values
            count += 1
    spio.savemat('concate_labels.mat', mdict={'Labels': obj_arr})
    return

def get_attempt_copulation():
    df = pd.read_excel('C:\\Users\hunte\OneDrive\Lab\Parameters_Determine\Behavior Transition\Code\DE_python\\attempt_corpulation.xlsx', header=0)
    length_list = []
    gap = []
    for condition in ['WT','Or47b']:
        for video_index in [x+1 for x in range(19)]:
            temp_df = df.loc[df['condition'] == condition]
            temp_df = temp_df.loc[temp_df['video_index'] == video_index]
            temp_df = temp_df.reset_index(drop=True)
            if temp_df.iloc[0]['mate'] == 'no':
                length_list.append(int(36000))
            elif temp_df.iloc[0]['mate'] == 'yes':
                length_list.append(int(temp_df.iloc[-1]['AC_begin']))
            for index, row in temp_df.iterrows():
                temp_temp_df = temp_df.dropna()
                if index + 2 <= temp_temp_df.shape[0]:
                    gap.append(int(temp_df.iloc[index+1]['AC_begin'] - row['AC_end']))
                else:
                    gap.append(math.nan)
    df.insert(loc=4,column='gap',value=gap)
    df = df.dropna(subset=['AC_begin'])

    WT_length =length_list[:int(len(length_list)/2)]
    Or47b_length = length_list[int(len(length_list)/2):]
    Or47b_length[6] = 35627

    WT_corpulation_stamp = []
    Or47b_corpulation_stamp = []

    length = df['AC_end'] - df['AC_begin']
    df.insert(loc=4,column='length',value=length)

    #################################################
    # df = df.loc[df['mate'] == 'no']
    # conditions = np.array((df['mate'] == 'yes',
    #                        df['success'] == 'yes'))
    # df = df.loc[np.logical_and.reduce(conditions)]
    #################################################


    wtdf = df.loc[df['condition'] == 'WT']
    or47bdf = df.loc[df['condition'] == 'Or47b']
    for index, row in wtdf.iterrows():
        index = int(row['video_index'] - 1)
        WT_corpulation_stamp.append(int(row['AC_begin'] + ([0] + np.cumsum(WT_length).tolist()[:-1])[index]))
    for index, row in or47bdf.iterrows():
        index = int(row['video_index'] - 1)
        Or47b_corpulation_stamp.append(int(row['AC_begin'] + ([0] + np.cumsum(Or47b_length).tolist()[:-1])[index]))
    obj_arr = np.zeros((1,2), dtype=np.object)
    obj_arr[0,0] = Or47b_corpulation_stamp
    obj_arr[0,1] = WT_corpulation_stamp
    spio.savemat('Corpulation_stamp.mat', mdict={'Corpulation_stamp': obj_arr})
    return

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    get_attempt_copulation()
    # get_gt_matlab(['chase','walk','sing','touch','stay_close','search','still'])
    exit()
    # Score_names = {'scores_chase_021218', 'scores_wing_search_021218', 'scores_wing_sing_021218', 'scores_touch_021218',
    #                'scores_stay_close_022818', 'scores_walk_030918', 'scores_still_030918'}
    # result = get_score_gt(extract_gt_label('wing_search','Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_wing_search_042718.txt'))
    result1 = extract_gt_label('still','Z:\#Yinan\Behavior Transition\GT\\3rd_batch\GT_still_042718.txt')
    result = get_score_gt(result1)
    result = result.loc[result['video_folder_path_abs'].str.contains('01_cam_')]
    # result = result.loc[(result['video_folder_path_abs'].str.contains('02_cam_20161019T113144')) | (result['video_folder_path_abs'].str.contains('01_cam_20161019T111429')) | (result['video_folder_path_abs'].str.contains('01_cam_20170721T142032'))]
    print(result.loc[result['GT'] == 0])
