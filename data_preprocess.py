import pandas as pd
import os

train_dir = "./data/hy_round1_train_20200102"
test_dir = "./data/hy_round1_testA_20200102"

test = pd.DataFrame(columns=['渔船ID', 'x', 'y', '速度', '方向', 'time'])

for root, dirs, files in os.walk(test_dir):
    for file in files:
        filename = os.path.join(root, file)
        temp = pd.read_csv(filename)
        test = pd.concat([test, temp])
print(test.info())

train = pd.DataFrame(columns=['渔船ID', 'x', 'y', '速度', '方向', 'time'])

for root, dirs, files in os.walk(train_dir):
    for file in files:
        filename = os.path.join(root, file)
        temp = pd.read_csv(filename)
        train = pd.concat([train, temp])
print(train.info())



test.to_csv("./data/test_origin.csv", index=None)
train.to_csv("./data/train_origin.csv", index=None)
