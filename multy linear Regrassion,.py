# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 23:14:17 2021

@author: masoo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

auto_mpg = pd.read_csv('C:/Users/masoo/Downloads/auto-mpgdata.csv', names=['mpg','cylinder','displacement','horsepower','weight','acceleration','model year','origin','car name']
, header=None)
auto_mpg.head()
auto_mpg.columns
#columns
1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous
5. weight: continuous
6. acceleration: continuous
7. model year: multi-valued discrete
8. origin: multi-valued discrete
9. car name: string (unique for each instance)

auto_mpg.describe()
auto_mpg.info()
auto_mpg.isnull().sum()
auto_mpg.horsepower.median()
auto_mpg.horsepower.unique()
auto_mpg.horsepower.fillna(auto_mpg.horsepower.median(), inplace = True)
auto_mpg.isnull().sum()
auto_mpg.drop('car name', inplace = True, axis = 1)

sns.boxplot(data = auto_mpg)
plt.xticks( rotation = 40)
plt.show()
# y = mpg
# x = 'mpg','cylinder','displacement','horsepower','weight','acceleration','model year','origin','car name

from sklearn.model_selection import train_test_split
#for multylinear regression we dont need to reshape the data
y = np.array(auto_mpg.mpg)
x = np.array(auto_mpg.iloc[:,1:])

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = 0.20, random_state = 123)

#model creation
from sklearn.linear_model import LinearRegression
mul_reg = LinearRegression()

mul_reg.fit(X = train_x, y = train_y)
#model equation

mul_reg.coef_
mul_reg.intercept_
#equation, y = [-0.3989072 ,  0.01918989,  0.00491233, -0.00700489,  0.24086977,0.75980713,  1.63987083]x + (-21.81)

#model accuracy
mul_reg.score(test_x, test_y) # 0.788143


 cylinders = [8,6,4,6]
 displacement= [500,390,440,414]
 horsepower= [90,110,56,70]
 weight= [3555,4321,3456,2345]
 acceleration= [10.5,11.2,5.6,7.9]
 model year= [70,73,75,70]
 origin= [1,1,2,3]
 
to_pred = {'cylinders':[8,6,4,6], 'displacement': [500,390,440,414], 'horsepower' : [90,110,56,70], 'weight': [3555,4321,3456,2345],
          'acceleration': [10.5,11.2,5.6,7.9], 'model year': [70,73,75,70], 'origin' : [1,1,2,3]}

to_predict = pd.DataFrame(to_pred)

#to predict new mpg value
mul_reg.predict(to_predict)






