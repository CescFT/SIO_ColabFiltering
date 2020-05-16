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
    
    '''
    ratings = np.full((n_users, n_items), np.nan)    
    

    for row in all_ratings.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]    
        
    sparsity = 100 - float(np.isnan(ratings[0]).sum())
    sparsity /= (ratings[1].shape[0])
    sparsity *= 100
    print("Coeficiente de sparseidad: {:4.2f}%".format(sparsity))
    
    user_ratings['array'] = ratings.tolist()
    user_ratings['std'] = user_ratings['array'].apply(lambda x: np.nanstd(x))
    user_ratings['visits'] = user_ratings['array'].apply(lambda x: 100 - float(np.isnan(x).sum()))
    
    user_df = user_ratings.copy()
    user_df.sort_values('std', inplace = True)
    
    rating_split = user_df.iloc[:7342,:]
    user_id_list = rating_split['userID']
    
    df_user = all_ratings.loc[user_id_list]
    df_user.reset_index(inplace=True, drop = True)
    
    ratings_train, ratings_test = train_test_split(df_user, test_size = 0.3, random_state=42)
    ratings_train.shape
    
    #sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_train)
    '''
    
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
    
    
except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")