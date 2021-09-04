
import numpy as np
import pandas as pd

df_all = pd.read_csv('C:/Users/navne/Desktop/Textbooks/Machine Learning/insurance.csv')

df=df_all[['age','bmi','children','charges']]
#print(df.head())

#x=df[['age','bmi','children']]
#y=df['charges']

#print(x.shape)


import random
a=np.random.permutation(1338)
#print(a)

# 20%....80% of train
size_train=[268,402,526,670,804,938,1072]

for i in size_train:
    train=df.loc[a[0:i]]
    test=df.loc[a[i:1338]]

    #print(train.head())
    #print(test.head())
    #x_0=
    x_train=train[['age','bmi','children']]
    y_train=train[['charges']]

    x_train_trans=x_train.T
    #print(type(x_train_trans))

    mult=x_train_trans.dot(x_train)
    inve=pd.DataFrame(np.linalg.pinv(mult.values), mult.columns, mult.index)
    theta_n=inve.dot(x_train_trans)
    theta=theta_n.dot(y_train)
    #print(theta.shape)

    '''
    sum=0
    for index,row in x_train.iterrows():
        print(type(row))
        sum=sum+(theta.T.dot(row)-y_train.loc[index])**2
    sum=sum/268
    '''

    train_pre=(((x_train.dot(theta)-y_train)**2).sum())/i
    #print(x_train.dot(theta)-y_train)
    #print(train_pre)


    x_test=test[['age','bmi','children']]
    y_test=test[['charges']]

    test_pre=(((x_test.dot(theta)-y_test)**2).sum())/(1338-i)
    print(test_pre)
