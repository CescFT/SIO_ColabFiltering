import psycopg2
import pandas as pd
import openpyxl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

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
    pivot_ratings = all_ratings.pivot_table(index=['restaurantID'], columns=['userID'], values='rating')

    data_csv = pd.read_csv('..\\..\\cluster_users.csv', sep=';')

    for i in range(1):
        cluster0_df = data_csv[data_csv['k-means'] == i].sort_values('userID')
        if cluster0_df.empty:
            continue
        columns_cluster0 = pd.Series(cluster0_df['userID']).values

        cluster0_users = pivot_ratings[columns_cluster0]

        cluster0_not_null = cluster0_users.notnull().all(axis=0)
        cluster0_not_null = pd.Series.to_frame(cluster0_not_null)
        cluster0_not_null.reset_index(inplace=True)
        cluster0_not_null2 = cluster0_not_null[cluster0_not_null[0]==True]['userID']
        cluster0_notnans = cluster0_users[cluster0_not_null2]
        # Hauriem de separar les dades que no son nules en dos grups per fer el split test.

        # cluster0_null = cluster0_not_null[cluster0_not_null[0]==False]['userID']
        # cluster0_nans = cluster0_users[cluster0_null]

        X_train, X_test, y_train, y_test = train_test_split(cluster0_notnans, cluster0_notnans, random_state=4)

        regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=30,
                                                                  random_state=0))

        print(X_train)
        print(y_train)
        # Fit on the train data
        regr_multirf.fit(X_train, y_train)

        # Check the prediction score
        score = regr_multirf.score(X_test, y_test)

        df_final = regr_multirf.predict(cluster0_notnans)
        print(df_final)


except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
