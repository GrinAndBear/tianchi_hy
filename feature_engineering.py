import pandas as pd


'''
创建一些统计特征
'''


def statistics_fea(df, feature_Name):
    feature = df[["渔船ID"]].drop_duplicates()
    df_x_max = df.groupby("渔船ID")[feature_Name].max().reset_index()
    df_x_max.rename(columns={feature_Name : feature_Name + "_max"}, inplace = True)

    df_x_min = df.groupby("渔船ID")[feature_Name].min().reset_index()
    df_x_min.rename(columns={feature_Name : feature_Name + "_min"}, inplace = True)

    df_x_std = df.groupby("渔船ID")[feature_Name].std().reset_index()
    df_x_std.rename(columns={feature_Name : feature_Name + "_std"}, inplace = True)

    df_x_median = df.groupby("渔船ID")[feature_Name].median().reset_index()
    df_x_median.rename(columns={feature_Name : feature_Name + "_median"}, inplace = True)

    df_x_mean = df.groupby("渔船ID")[feature_Name].mean().reset_index()
    df_x_mean.rename(columns={feature_Name : feature_Name + "_mean"}, inplace = True)

    feature = pd.merge(feature, df_x_max, on="渔船ID", how="left")
    feature = pd.merge(feature, df_x_min, on="渔船ID", how="left")
    feature = pd.merge(feature, df_x_std, on="渔船ID", how="left")
    feature = pd.merge(feature, df_x_median, on="渔船ID", how="left")
    feature = pd.merge(feature, df_x_mean, on="渔船ID", how="left")

    return feature

if __name__ == "__main__":
    '''
    create statistics feature
    '''
    test_origin = pd.read_csv("./data/test_origin.csv")
    feature = test_origin[["渔船ID"]].drop_duplicates()
    x = statistics_fea(test_origin, "x")
    y = statistics_fea(test_origin, "y")
    speed = statistics_fea(test_origin, "速度")
    direction = statistics_fea(test_origin, "方向")

    feature = pd.merge(feature, x, on="渔船ID", how="left")
    feature = pd.merge(feature, y, on="渔船ID", how="left")
    feature = pd.merge(feature, speed, on="渔船ID", how="left")
    feature = pd.merge(feature, direction, on="渔船ID", how="left")
    feature.to_csv("./data/test.csv", index = None)
    print(feature.info())

    train_origin = pd.read_csv("./data/train_origin.csv")
    feature = train_origin[["渔船ID", "type"]].drop_duplicates()
    label_dict = {"刺网":0, "围网":1, "拖网":2}
    feature["type"] = feature["type"].apply(lambda x : label_dict[x])
    x = statistics_fea(train_origin, "x")
    y = statistics_fea(train_origin, "y")
    speed = statistics_fea(train_origin, "速度")
    direction = statistics_fea(train_origin, "方向")
    feature = pd.merge(feature, x, on="渔船ID", how="left")
    feature = pd.merge(feature, y, on="渔船ID", how="left")
    feature = pd.merge(feature, speed, on="渔船ID", how="left")
    feature = pd.merge(feature, direction, on="渔船ID", how="left")
    feature.to_csv("./data/train.csv", index = None)
    print(feature.info())
