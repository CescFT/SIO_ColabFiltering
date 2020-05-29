# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:28:01 2020

@author: Sancho
"""


import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
import psycopg2
import re

try:
    connection = psycopg2.connect(user="postgres",
                                  password="aleixDatabase",
                                  host="localhost",
                                  port="3306",
                                  database="siodb")

    cursor = connection.cursor()

    median_query = '''SELECT username, restaurant, rating FROM ratings'''

    cursor.execute(median_query)
    ratings = cursor.fetchall()

    all_ratings = pd.DataFrame(ratings, columns=['userID', 'restaurantID', 'rating'])
    all_ratings['index'] = all_ratings['userID'].copy()
    all_ratings.set_index('index', inplace = True)
    
    user_ratings = all_ratings.drop(['restaurantID'], axis=1)
    user_ratings = user_ratings.groupby('userID', as_index=False).mean()
    
    n_users = all_ratings.userID.unique().shape[0]
    n_users
    
    n_items = all_ratings.restaurantID.unique().shape[0]
    n_items
    
    filter_users = all_ratings['userID'].value_counts() > 90
    filter_users = filter_users[filter_users].index.tolist()
    
    filter_restaurants = all_ratings['restaurantID'].value_counts() > 68000
    filter_restaurants = filter_restaurants[filter_restaurants].index.tolist()
    
    df_new = all_ratings[(all_ratings['userID'].isin(filter_users)) & (all_ratings['restaurantID'].isin(filter_restaurants))]
    print('The original data frame shape:\t{}'.format(all_ratings.shape))
    print('The new data frame shape:\t{}'.format(df_new.shape))
    
    reader = Reader(rating_scale=(-10, 10))
    data = Dataset.load_from_df(df_new[['userID', 'restaurantID', 'rating']], reader)
    
    benchmark = []
    # Iterate over all algorithms
    for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
        # Perform cross validation
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
        
    surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    
    
    all_data = Dataset.load_from_df(all_ratings[['userID', 'restaurantID', 'rating']], reader)
    trainset, testset = train_test_split(all_data, test_size=0.25) 
    algo = SVD()
    predictions = algo.fit(trainset).test(testset)
    mae = accuracy.mae(predictions)
    mae
    
    
    sampledf = pd.read_csv('..\..\PlantillaPrediccions.csv', sep = ';', names = ['userID', 'restaurantID', 'rating'])
    sampledf_all = sampledf.copy()
    sampledf['userID'] = sampledf['userID'].apply(lambda x: re.findall('\d+', x )[0])
    sampledf['restaurantID'] = sampledf['restaurantID'].apply(lambda x: re.findall('\d+', x )[0])
    
    sample_list = sampledf.drop('rating', axis = 1)
    
    pred_list = []
    
    for uid, iid in sample_list.itertuples(index=False):
        pred = algo.predict(int(uid), int(iid), r_ui=0, verbose=True)
        pred_dict = {'userID' : uid, 'restaurantID' : iid, 'rating' : format(float(pred.est), '.4f')}
        pred_list.append(pred_dict)
        
    final_df = pd.DataFrame(pred_list)
       
    final_df['userID'] = sampledf_all['userID']
    final_df['restaurantID'] = sampledf_all['restaurantID']  

    final_df.to_csv('..\\..\\tokenName5.csv', sep = ';', header = False, index = False)      
    
except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")