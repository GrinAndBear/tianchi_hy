import pandas as pd
import lightgbm as lgb

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# train, valid = train[:5000], train[5000:]

train_X = train.drop(["渔船ID", "type"], axis = 1)
train_y = train["type"]
train_X, train_y = train_X.values, train_y.values

# valid_X = train.drop(["渔船ID", "type"], axis = 1)
# valid_y = train["type"]
# train_X, train_y = train_X.values, train_y.values
#


pred_test = test[["渔船ID"]]

test_X = test.drop("渔船ID", axis = 1)
test_X = test_X.values


params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_leaves': 32,
    'min_data_in_leaf': 100,
    'learning_rate': 0.06,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.4,
    'lambda_l2': 0.5,
    'min_gain_to_split': 0.2,
    'verbose': -1,
}
print('Training...')
trn_data = lgb.Dataset(train_X, train_y)
test_data = lgb.Dataset(test_X)
clf = lgb.train(params,
                trn_data,
                num_boost_round = 1000)
                # valid_sets = [trn_data,val_data],
                # verbose_eval = 100,
                # early_stopping_rounds = 100)
print('Predicting...')

y_prob = clf.predict(test_X, num_iteration=clf.best_iteration)
y_pred = [list(x).index(max(x)) for x in y_prob]
# print("AUC score: {:<8.5f}".format(metrics.accuracy_score(y_pred, test_y)))

pred_test["type"] = y_pred
label_dict = {0 : "刺网",1 : "围网",2 : "拖网"}
pred_test["type"] = pred_test["type"].apply(lambda x : label_dict[x])
pred_test.sort_values("渔船ID", inplace=True)

pred_test.to_csv("./submit/pred_010411.csv", index = None, encoding = "utf8", header = False)