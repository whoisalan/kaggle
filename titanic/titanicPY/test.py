import numpy as np
import pandas as pd



data_train=pd.read_csv("train.csv")
age_df = data_train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]  # 只保留匹配的
# print(age_df)
# known_age = age_df[age_df.Age.notnull()].as_matrix()
# print(known_age[:,0]) # 所有列中的第0个
# print(known_age[:, 1::])

print(data_train.Cabin)
data_train.loc[data_train.Cabin.notnull()] = 1
data_train.loc[data_train.Cabin.isnull()] = 0
print(data_train.Cabin)
# data_train.loc[(data_train.Cabin.isnull()), 'Cabin'] = "No"

# unknown_age = age_df[age_df.Age.isnull()].as_matrix()
# print(known_age)
