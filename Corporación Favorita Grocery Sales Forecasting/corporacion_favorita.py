#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:06:45 2017
@author: FPCarneiro
"""

MODELDIR = "model/"
defaul_group_by = ['store_nbr','item_nbr']

import gc
import pandas as pd
import tensorflow as tf
#from sklearn.model_selection import train_test_split

import preprocessing as pp
import train_tensor_flow as ttf

import imp
imp.reload(pp)

train, test = pp.load_train_test_set(train_year=2017)
train = pp.adjust_unit_sales(train)

#train = train.loc[train.date>=pd.datetime(2017,6,1)]

ma_dw = pp.get_calculated_column(train[['item_nbr','store_nbr','dow','unit_sales']], 
                                       group_by = defaul_group_by + ['dow'], new_column_name = 'madw').reset_index()
                                        
ma_wk = pp.get_calculated_column(ma_dw[['item_nbr','store_nbr','madw']], target_column = 'madw', new_column_name = 'mawk').reset_index()                                        

train = train.set_index(["store_nbr", "item_nbr", "date"])
train = train.sort_index()
#train = train.reset_index()
grouped = train.groupby(level=['store_nbr', 'item_nbr'])

train.drop('dow', 1, inplace=True)
 
train = pp.fullfill_dataset(train)
lastdate = train.iloc[train.shape[0]-1].date
        
ma_is = pp.get_calculated_column(train[['item_nbr','store_nbr','unit_sales']], new_column_name = 'mais')

for i in [112,56,28,14,7,3,1]:
    ma_is = ma_is.join(pp.get_mean(train, lastdate, i), how='left')

ma_is['mais'] = ma_is.median(axis=1)
ma_is.reset_index(inplace = True)
ma_is.drop(list(ma_is.columns.values)[3:],1,inplace=True)

test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])


test['unit_sales'] = test.mais 
pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]

test.loc[pos_idx, 'unit_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']
test.loc[:, "unit_sales"].fillna(0, inplace=True)
test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1)

test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.5

test[['id','unit_sales']].to_csv('ma8dwof.csv.gz', index=False, float_format='%.3f', compression='gzip')















stores_holidays = pp.consolidate_holidays(stores, holidays_events)

train = pp.adjust_unit_sales(train)

train = pp.attach_stores(train, stores)
train = pp.attach_holidays_in_stores(train, stores_holidays)
train = pp.attach_items(train, items)

test = pp.attach_stores(test, stores)
test = pp.attach_holidays_in_stores(test, stores_holidays)
test = pp.attach_items(test, items)

del stores, holidays_events, items

gc.collect()

x = train.drop(['unit_sales', 'date', 'description', 'locale', 'year'], axis = 1)
y = train['unit_sales']

del train
gc.collect()

x_eval = test.drop(['date', 'description', 'locale', 'year', 'id'], axis = 1)

# TENSOR FLOW LINEAR REGRESSION ###############################################################################################

base_columns = ttf.get_base_columns()

model = tf.estimator.LinearRegressor(model_dir=ttf.MODELDIR, feature_columns = base_columns)

input_func = tf.estimator.inputs.pandas_input_fn(x = x, y = y, batch_size = 1024, num_epochs = None, shuffle = True)
train_input_func = tf.estimator.inputs.pandas_input_fn(x = x, y = y, batch_size = 1024, num_epochs = 1000, shuffle = False)
predict_input_func = tf.estimator.inputs.pandas_input_fn(x = x_eval, shuffle = False)

model.train(input_fn=input_func, steps = 1000)

train_metrics = model.evaluate(input_fn = train_input_func, steps = 1000)

predict_results = model.predict(input_fn = predict_input_func)

pred = [pd.np.expm1(i['predictions'][0]) for i in predict_results]

se = pd.Series(pred)
test['unit_sales'] = se.values


test[['id','unit_sales']].to_csv('sub0.csv.gz', index=False, float_format='%.3f', compression='gzip')

#50% more for promotion items
test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.5  
        
test[['id','unit_sales']].to_csv('sub1.csv.gz', index=False, float_format='%.3f', compression='gzip')