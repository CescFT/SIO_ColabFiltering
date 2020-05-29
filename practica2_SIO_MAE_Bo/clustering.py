import psycopg2
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

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

    user_ratings = all_ratings.drop(['restaurantID'], axis=1)
    user_ratings = user_ratings.groupby('userID', as_index=False).mean()

    user_std = all_ratings.drop(['restaurantID'], axis=1)
    user_std = user_std.groupby('userID', as_index=False).std()

    user_visits = all_ratings.drop(['rating'], axis=1)
    user_visits = user_visits.groupby('userID', as_index=False).count().rename(columns={'restaurantID': 'visit'})

    restaurant_ratings = all_ratings.drop(['userID'], axis=1)
    restaurant_ratings = restaurant_ratings.groupby('restaurantID', as_index=False).mean()

    restaurant_median = all_ratings.drop(['userID'], axis=1)
    restaurant_median = restaurant_median.groupby('restaurantID', as_index=False).median()

    restaurant_visits = all_ratings.drop(['rating'], axis=1)
    restaurant_visits = restaurant_visits.groupby('restaurantID', as_index=False).count().rename(
        columns={'userID': 'visit'})

    user_rest = all_ratings.copy()
    user_rest['user_mean'] = user_rest.groupby('userID')['rating'].transform('mean')
    rest_user = user_rest.groupby('restaurantID')['user_mean'].mean()
    rest_user.reset_index(inplace=True, drop=True)
    print(rest_user)

    X = pd.DataFrame(columns=['mean_user', 'std_user'])
    X['mean_user'] = user_ratings['rating']
    X['std_user'] = user_std['rating']
    print(X)

    km = KMeans(
        n_clusters=100, init='k-means++',
        n_init=10, max_iter=500, random_state=0
    )
    y_km = km.fit_predict(X)
    cluster_rest = X.copy()
    cluster_rest['k-means'] = y_km
    cluster_rest['userID'] = user_ratings['userID']
    cluster_rest = cluster_rest.sort_values('mean_user')
    print(cluster_rest)
    cluster_rest.to_csv('C:\\Users\\Sancho\\Desktop\\cluster_restaurants.csv', index=False)
    cluster_rest_cont = cluster_rest.groupby('k-means').count()
    print(cluster_rest_cont)
    print(max(pd.Series(cluster_rest_cont).values))
    print(min(pd.Series(cluster_rest_cont).values))
except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
