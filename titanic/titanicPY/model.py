mpl.rcParams['font.sans-serif'] = ['SimHei']

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from pylab import *

"""
初识数据
"""
data_train=pd.read_csv("train.csv")
data_test =pd.read_csv("test.csv")
    #返回的是DataFrame
# data_train.info()
    #返回一个DataFrame的简要信息
# print(data_train.describe())
    #返回一个数据的汇总统计





"""
可视化不同属性的分布
"""
# plt.subplot2grid((2,3),(0,0))
# data_train.Survived.value_counts().plot(kind='bar')
# plt.title("获救情况")
# plt.ylabel("人数")
#
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.ylabel("人数")
# plt.title("乘客登记分布")
#
#plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# # 散点图
# plt.ylabel("年龄")
# plt.grid(b=True,which="major",axis='y')
# plt.title("年龄|获救分布")
#
# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel("年龄")
# plt.ylabel("密度")
# plt.title("各等级的乘客年龄分布")
# plt.legend(('头等舱','2等舱','3等舱'),loc='best')
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title("各登船口岸上船人数")
# plt.ylabel("人数")
# plt.show()

"""
可视化不同属性与获救结果的关系
"""
# #船舱等级与存活的关系
# Survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived==1].value_counts()
# num = Survived_1+Survived_0
# df = pd.DataFrame({"获救":Survived_1/num,"未获救":Survived_0/num})
# df.plot(kind="bar",stacked = True)
# plt.title("各个舱位获救情况")
# plt.ylabel("人数")
# plt.xlabel("舱位等级")
# plt.grid(b=True,axis="y")
# plt.show()

# #性别与存活的关系
# Survived_0 = data_train.Sex[data_train.Survived==0].value_counts()
# Survived_1 = data_train.Sex[data_train.Survived==1].value_counts()
# num = Survived_1+Survived_0
# df = pd.DataFrame({"获救":Survived_1/num,"未获救":Survived_0/num})
# df.plot(kind="bar",stacked = True)
# plt.title("性别影响")
# plt.xlabel("性别")
# plt.grid(b=True,axis="y")
# plt.ylabel("人数")
# plt.show()

#登舱口存活的关系
# Survived_0 = data_train.Embarked[data_train.Survived==0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived==1].value_counts()
# num = Survived_1+Survived_0
# df = pd.DataFrame({"获救":Survived_1,"未获救":Survived_0})
# df.plot(kind="bar",stacked = True)
# plt.title("登舱口影响")
# plt.xlabel("登舱口")
# plt.grid(b=True,axis="y")
# plt.ylabel("人数")
# plt.show()

#表兄妹个数影响
# Survived_0 = data_train.SibSp[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.SibSp[data_train.Survived == 1].value_counts()
# num = Survived_0+Survived_1
# df = pd.DataFrame({"获救":Survived_1/num,"未获救":Survived_0/num})
# df.plot(kind = "bar",stacked = True)
# plt.title("表兄妹个数影响")
# plt.xlabel("个数")
# plt.ylabel("人数")
# plt.show()


# 船票费用
# mean = data_train.Fare.mean()
# Survived_h = data_train[data_train.Fare>mean].Survived[data_train.Survived==1].value_counts()
# Survived_l = data_train[data_train.Fare<=mean].Survived[data_train.Survived==1].value_counts()
# num1 = data_train.PassengerId[data_train.Fare>mean].count()
# num2 = data_train.PassengerId[data_train.Fare<=mean].count()
# df = pd.DataFrame({"有钱":Survived_h/num1,"没钱":Survived_l/num2})
# df.plot(kind = 'bar')
# plt.ylabel("人数")
# plt.xlabel("fare")
# plt.show()


"""
特征工程
"""
# print(data_train.Age.describe())


# 使用随机森林对缺失数据进行回归
def set_missing_age(df):
    # 已知的数值特征
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    # 将乘客分为已知年龄与未知年龄
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:,0]
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # random_state--->int--->随机数生成seed
    # random_state--->instance--->随机数生成器
    # n_estimators--->想要建立子树的数量。
    # n_jpbs--->这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。

    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # [start:end:step]
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 1
    df.loc[(df.Cabin.isnull()),'Cabin'] = 0
    return df




data_train, rfr = set_missing_age(data_train)
data_train = set_Cabin_type(data_train)

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
data_test = set_Cabin_type(data_test)

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age' ] = predictedAges


dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

scaler1 = preprocessing.StandardScaler()
age_scale_param = scaler1.fit(df_test['Age'])
df_test['Age_scaled'] = scaler1.fit_transform(df_test['Age'], age_scale_param)
fare_scale_param = scaler1.fit(df_test['Fare'])
df_test['Fare_scaled'] = scaler1.fit_transform(df_test['Fare'], fare_scale_param)


train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')


train_np = train_df.as_matrix()


y = train_np[:,0]
X = train_np[:,1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X,y)




test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
prediction=clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':prediction.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv",index=False)


