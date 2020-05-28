# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:28:01 2020

@author: Sancho
"""


import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import cosine
import psycopg2
from scipy import sparse
import sklearn.metrics.pairwise as pw

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
    
    user_ratings_pos = user_ratings[user_ratings['rating'] > 0].reset_index(drop = True)
    all_ratings_pos = all_ratings[all_ratings['userID'].isin(user_ratings_pos['userID'])]
    
    pivot_ratings = all_ratings_pos.pivot_table(index=['userID'], columns=['restaurantID'], values='rating')
    
    reduced_pivot = pivot_ratings.copy()
    
    nan_pivot = reduced_pivot.fillna(0)
    
    matrix = sparse.csr_matrix(nan_pivot)
    
    sim_matrix = 1 - pw.cosine_distances(matrix)
    
except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")