# -*- coding: utf-8 -*

'''
@author: PY131

@thoughts:  as the samples are extremely imbalance (N/P ratio ~ 1.2k),
            here we use sub-sample on negative samples.
            1-st: using k_means to make clustering on negative samples (clusters_number ~ 1k)
            2-nd: subsample on each clusters based on the same ratio,
                  the ratio was selected to be the best by testing in random sub_sample + GBDT
            3-rd: selecting the best parameter for GBDT classifier
            4-th: using GBDT model for training and predicting on sub_sample set.

            here is 2-nd to 4-th step
'''
# depending package
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import time

########## file path ##########
##### input file

# training set keys uic-label with k_means clusters' label
path_df_part_1_uic_label_cluster = "./data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "./data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic = "./data/mobile/raw/df_part_3_uic.csv"

# data_set features
path_df_part_1_U = "./data/mobile/feature/df_part_1_U.csv"
path_df_part_1_I = "./data/mobile/feature/df_part_1_I.csv"
path_df_part_1_C = "./data/mobile/feature/df_part_1_C.csv"
path_df_part_1_IC = "./data/mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI = "./data/mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC = "./data/mobile/feature/df_part_1_UC.csv"

path_df_part_2_U = "./data/mobile/feature/df_part_2_U.csv"
path_df_part_2_I = "./data/mobile/feature/df_part_2_I.csv"
path_df_part_2_C = "./data/mobile/feature/df_part_2_C.csv"
path_df_part_2_IC = "./data/mobile/feature/df_part_2_IC.csv"
path_df_part_2_UI = "./data/mobile/feature/df_part_2_UI.csv"
path_df_part_2_UC = "./data/mobile/feature/df_part_2_UC.csv"

path_df_part_3_U = "./data/mobile/feature/df_part_3_U.csv"
path_df_part_3_I = "./data/mobile/feature/df_part_3_I.csv"
path_df_part_3_C = "./data/mobile/feature/df_part_3_C.csv"
path_df_part_3_IC = "./data/mobile/feature/df_part_3_IC.csv"
path_df_part_3_UI = "./data/mobile/feature/df_part_3_UI.csv"
path_df_part_3_UC = "./data/mobile/feature/df_part_3_UC.csv"

# item_sub_set P
path_df_P = "./data/raw/tianchi_fresh_comp_train_item.csv"

##### output file
path_df_result = "./res_xgb_k_means_subsample.csv"
path_df_result_tmp = "./xgb_temp.csv"




# some functions
def df_read(path, mode='r'):
    '''the definition of dataframe loading function
    '''
    data_file = open(path, mode)
    try:
        df = pd.read_csv(data_file, index_col=False)
    finally:
        data_file.close()
    return df


def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size

    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df):
        return df
    else:
        return df.sample(n=sub_size)


##### loading data of part 1 & 2
df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)
df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)

df_part_1_U = df_read(path_df_part_1_U)
df_part_1_I = df_read(path_df_part_1_I)
df_part_1_C = df_read(path_df_part_1_C)
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

df_part_2_U = df_read(path_df_part_2_U)
df_part_2_I = df_read(path_df_part_2_I)
df_part_2_C = df_read(path_df_part_2_C)
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)


##### generation of training set & valid set
def train_set_construct(np_ratio=1, sub_ratio=1):
    '''
    # generation of train set
    @param np_ratio: int, the sub-sample rate of training set for N/P balanced.
    @param sub_ratio: float ~ (0~1], the further sub-sample rate of training set after N/P balanced.
    '''
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)

    frac_ratio = sub_ratio * np_ratio / 1200
    for i in range(1, 1001, 1):
        train_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac=frac_ratio)
        train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])

        train_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac=frac_ratio)
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    print("training subset uic_label keys is selected.")

    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I, how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C, how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I, how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C, how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)

    # using all the features for training gbdt model
    train_X = train_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    train_y = train_df['label'].values
    print("train subset is generated.")
    return train_X, train_y


def valid_set_construct(sub_ratio=0.1):
    '''
    # generation of valid set
    @param sub_ratio: float ~ (0~1], the sub-sample rate of original valid set
    '''
    valid_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)
    valid_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)

    for i in range(1, 1001, 1):
        valid_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac=sub_ratio)
        valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])

        valid_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac=sub_ratio)
        valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])

    # constructing valid set
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I, how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C, how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I, how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C, how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)

    # using all the features for valid gbdt model
    valid_X = valid_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    valid_y = valid_df['label'].values
    print("valid subset is generated.")

    return valid_X, valid_y


def valid_train_set_construct(valid_ratio=0.5, valid_sub_ratio=0.5, train_np_ratio=1, train_sub_ratio=0.5):
    '''
    # generation of train set
    @param valid_ratio: float ~ [0~1], the valid set ratio in total set and the rest is train set
    @param valid_sub_ratio: float ~ (0~1), random sample ratio of valid set
    @param train_np_ratio:(1~1200), the sub-sample ratio of training set for N/P balanced.
    @param train_sub_ratio: float ~ (0~1), random sample ratio of train set after N/P subsample

    @return valid_X, valid_y, train_X, train_y
    '''
    msk_1 = np.random.rand(len(df_part_1_uic_label_cluster)) < valid_ratio
    msk_2 = np.random.rand(len(df_part_2_uic_label_cluster)) < valid_ratio

    valid_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[msk_1]
    valid_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[msk_2]

    valid_part_1_uic_label = valid_df_part_1_uic_label_cluster[valid_df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=valid_sub_ratio)
    valid_part_2_uic_label = valid_df_part_2_uic_label_cluster[valid_df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=valid_sub_ratio)

    ### constructing valid set
    for i in range(1, 1001, 1):
        valid_part_1_uic_label_0_i = valid_df_part_1_uic_label_cluster[valid_df_part_1_uic_label_cluster['class'] == i]
        if len(valid_part_1_uic_label_0_i) != 0:
            valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac=valid_sub_ratio)
            valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])

        valid_part_2_uic_label_0_i = valid_df_part_2_uic_label_cluster[valid_df_part_2_uic_label_cluster['class'] == i]
        if len(valid_part_2_uic_label_0_i) != 0:
            valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac=valid_sub_ratio)
            valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])

    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I, how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C, how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I, how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C, how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)

    # using all the features for valid rf model
    valid_X = valid_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    valid_y = valid_df['label'].values
    print("valid subset is generated.")

    ### constructing training set
    train_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[~msk_1]
    train_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[~msk_2]

    train_part_1_uic_label = train_df_part_1_uic_label_cluster[train_df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=train_sub_ratio)
    train_part_2_uic_label = train_df_part_2_uic_label_cluster[train_df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=train_sub_ratio)

    frac_ratio = train_sub_ratio * train_np_ratio / 1200
    for i in range(1, 1001, 1):
        train_part_1_uic_label_0_i = train_df_part_1_uic_label_cluster[train_df_part_1_uic_label_cluster['class'] == i]
        if len(train_part_1_uic_label_0_i) != 0:
            train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac=frac_ratio)
            train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])

        train_part_2_uic_label_0_i = train_df_part_2_uic_label_cluster[train_df_part_2_uic_label_cluster['class'] == i]
        if len(train_part_2_uic_label_0_i) != 0:
            train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac=frac_ratio)
            train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])

    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I, how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C, how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I, how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C, how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)

    # using all the features for training rf model
    train_X = train_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    train_y = train_df['label'].values
    print("train subset is generated.")

    return valid_X, valid_y, train_X, train_y
