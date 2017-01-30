# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:08:58 2017

@author: Admin
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Load dataframe
os.chdir(r'C:\Users\Admin\Desktop\clasif\code')
df = pd.read_pickle('datadf.pkl')
os.chdir(r'D:\Agus\HiddenProject')

#%% filter dataframe

df = df.query('c != "m.shmoos"')

count = Counter(df.c.values)
print(count)
mx = np.min(list(count.values()))

selected = []
for label in count.keys():
    ndxs = df[df['c'] == label].index.tolist()
    np.random.shuffle(ndxs)
    selected.extend(ndxs[:mx])

print(len(selected))
df = df.loc[selected]

count = Counter(df.c.values)
print(count)

#%% split dataframe

train, test_both = train_test_split(df, test_size=0.4)
test_one, test_two = train_test_split(test_both, test_size=0.5)

#%% Save data sets

train.to_pickle('Train.pandas')
test_one.to_pickle('First_Test.pandas')
test_two.to_pickle('Final_Test.pandas')