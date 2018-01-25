#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:06:45 2017
@author: FPCarneiro
"""

MODELDIR = "model/"

#import pandas as pd
import tensorflow as tf

def create_numeric_columns(column_name, column_type=tf.float32):
    return tf.feature_column.numeric_column(key = column_name, dtype = column_type)

def get_numeric_columns(columns):
    return [create_numeric_columns(c) for c in columns]

def get_base_numeric_columns(dataset, types_included):
    c = list(dataset.select_dtypes(include = types_included).columns)
    return get_numeric_columns(c)

def get_categorical_columns():
    categorical_list = list()
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('type', ['A', 'B', 'C', 'D', 'E']))
    
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('Additional', [0,1], dtype=tf.int8))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('Bridge', [0,1], dtype=tf.int8))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('Event', [0,1], dtype=tf.int8))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('Holiday', [0,1], dtype=tf.int8))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('Work_Day', [0,1], dtype=tf.int8))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('onpromotion', [0,1], dtype=tf.int8))

    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list(key='cluster', vocabulary_list=list(range(1,18)), dtype=tf.int32))
    categorical_list.append(tf.feature_column.categorical_column_with_hash_bucket('city', hash_bucket_size=22))
    categorical_list.append(tf.feature_column.categorical_column_with_hash_bucket('state', hash_bucket_size=16))
    categorical_list.append(tf.feature_column.categorical_column_with_hash_bucket('store_nbr', hash_bucket_size=54, dtype=tf.int8))
    categorical_list.append(tf.feature_column.categorical_column_with_hash_bucket('item_nbr', hash_bucket_size=4100, dtype=tf.int32))

#    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list(key='year', vocabulary_list=[2016, 2017], dtype=tf.int32))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list(key='month', vocabulary_list=list(range(1,13)), dtype=tf.int32))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list(key='day', vocabulary_list=list(range(1,32)), dtype=tf.int32))
    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list(key='dow', vocabulary_list=list(range(7)), dtype=tf.int32))

    categorical_list.append(tf.feature_column.categorical_column_with_vocabulary_list('perishable', [0,1]))
    categorical_list.append(tf.feature_column.categorical_column_with_hash_bucket('class', hash_bucket_size=335, dtype=tf.int32))
    categorical_list.append(tf.feature_column.categorical_column_with_hash_bucket('family', hash_bucket_size=35))
    return categorical_list

#def get_crossed_columns(list_columns, bsize):
#    return [tf.feature_column.crossed_column(list_columns, hash_bucket_size=bsize)]

def get_base_columns():
    return get_categorical_columns()