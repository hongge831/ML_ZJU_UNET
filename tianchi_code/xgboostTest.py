from xgboostHeader import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio=0.2,
#                                                                      valid_sub_ratio=1,
#                                                                      train_np_ratio=np_ratio,
#                                                                      train_sub_ratio=1)

train_X, train_y = train_set_construct(np_ratio=100, sub_ratio=1)

clf = XGBClassifier(
    silent=1,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    # nthread=4,# cpu 线程数 默认最大
    learning_rate=0.2,  # 如同学习率
    min_child_weight=1,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    max_depth=8,  # 构建树的深度，越大越容易过拟合
    gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    subsample=1,  # 随机采样训练样本 训练实例的子采样比
    max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=1,  # 生成树时进行的列采样
    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    # reg_alpha=0, # L1 正则项参数
    # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    objective= 'binary:logistic', #多分类的问题 指定学习任务和相应的学习目标
    # num_class=10, # 类别数，多分类与 multisoftmax 并用
    n_estimators=120,  # 树的个数
    seed=1000  # 随机种子
    # eval_metric= 'auc'
    )
clf.fit(train_X, train_y, eval_metric='auc')
# 设置验证集合 verbose=False不打印过程
# clf.fit(X_train, train_y, eval_set=[(X_train, train_y), (valid_X, valid_y)], eval_metric='auc', verbose=False)
# 获取验证集合结果
# evals_result = clf.evals_result()
# predict_y = clf.predict(valid_X)
# print(predict_y)
# print("Accuracy : %.2f" % metrics.roc_auc_score(valid_y, predict_y))
# print("f1_score: %.2f" % metrics.f1_score(valid_y, predict_y))


df_part_3_U  = df_read(path_df_part_3_U )
df_part_3_I  = df_read(path_df_part_3_I )
df_part_3_C  = df_read(path_df_part_3_C )
df_part_3_IC = df_read(path_df_part_3_IC)
df_part_3_UI = df_read(path_df_part_3_UI)
df_part_3_UC = df_read(path_df_part_3_UC)
batch = 0
for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize=100000):
    try:
        # construct of prediction sample set
        pred_df = pd.merge(pred_uic, df_part_3_U, how='left', on=['user_id'])
        pred_df = pd.merge(pred_df, df_part_3_I, how='left', on=['item_id'])
        pred_df = pd.merge(pred_df, df_part_3_C, how='left', on=['item_category'])
        pred_df = pd.merge(pred_df, df_part_3_IC, how='left', on=['item_id', 'item_category'])
        pred_df = pd.merge(pred_df, df_part_3_UI, how='left', on=['user_id', 'item_id', 'item_category'])
        pred_df = pd.merge(pred_df, df_part_3_UC, how='left', on=['user_id', 'item_category'])

        # fill the missing value as -1 (missing value are time features)
        pred_df.fillna(-1, inplace=True)

        # using all the features for training RF model
        pred_X = pred_df.as_matrix(
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

        # predicting
        # pred_y = (GBDT_clf.predict_proba(pred_X)[:, 1] > 0.45).astype(int)
        pred_y = clf.predict(pred_X)

        # generation of U-I pairs those predicted to buy
        pred_df['pred_label'] = pred_y
        # add to result csv
        pred_df[pred_df['pred_label'] == 1].to_csv(path_df_result_tmp,
                                                   columns=['user_id', 'item_id'],
                                                   index=False, header=False, mode='a')

        batch += 1
        print("prediction chunk %d done." % batch)

    except StopIteration:
        print("prediction finished.")
        break

#######################################################################
'''Step 3: generation result on items' sub set P 
'''

# loading data
df_P = df_read(path_df_P)
df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
df_pred = pd.read_csv(open(path_df_result_tmp,'r'), index_col=False, header=None)
df_pred.columns = ['user_id', 'item_id']

# output result
df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
df_pred_P.to_csv(path_df_result, index=False)


print(' - PY131 - ')

