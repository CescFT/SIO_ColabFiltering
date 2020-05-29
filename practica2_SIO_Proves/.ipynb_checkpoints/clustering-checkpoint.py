import psycopg2
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def groups(x):
    group_count = 7342
    if x < group_count:
        return 9
    elif x < group_count * 2:
        return 8
    elif x < group_count * 3:
        return 7
    elif x < group_count * 4:
        return 6
    elif x < group_count * 5:
        return 5
    elif x < group_count * 6:
        return 4
    elif x < group_count * 7:
        return 3
    elif x < group_count * 8:
        return 2
    elif x < group_count * 9:
        return 1
    else:
        return 0


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
    # print(user_ratings)

    user_visits = all_ratings.drop(['rating'], axis=1)
    user_visits = user_visits.groupby('userID', as_index=False).count().rename(columns={'restaurantID': 'visit'})
    # print(user_visits)

    user_visits_group = user_visits.copy()
    user_visits_group = user_visits_group.sort_values('visit')
    print(user_visits_group)

    user_visits_group.reset_index(inplace=True)
    user_visits_group.drop(['index'], axis=1, inplace=True)
    user_visits_group.reset_index(inplace=True)

    user_visits_group['group'] = user_visits_group['index'].apply(groups)
    user_visits_group.drop(['index'], axis=1, inplace=True)
    user_visits_group.sort_values('userID', inplace=True)
    print(user_visits_group)

    user_group_mean = user_visits_group.copy()
    user_group_mean = user_group_mean.drop(['visit'], axis=1)
    user_group_mean['rating'] = user_ratings['rating']

    '''
    user_rating_visitis = user_ratings.copy()
    user_rating_visitis['visit'] = user_visits['visit']
    user_rating_visitis = user_rating_visitis.set_index(['userID'])
    print(user_rating_visitis)

    restaurant_ratings = all_ratings.drop(['userID'], axis=1)
    restaurant_ratings = restaurant_ratings.groupby('restaurantID', as_index=False).mean()
    # print(restaurant_ratings)

    restaurant_visits = all_ratings.drop(['rating'], axis=1)
    restaurant_visits = restaurant_visits.groupby('restaurantID', as_index=False).count().rename(
        columns={'userID': 'visit'})
    # print(restaurant_visits)

    restaurant_ratings_sorted = pd.DataFrame(columns=['rating', 'visit'])
    restaurant_ratings_sorted['rating'] = restaurant_ratings['rating']
    restaurant_ratings_sorted['visit'] = restaurant_visits['visit']
    restaurant_ratings_sorted = restaurant_ratings_sorted.sort_values(by=['rating'])
    # print(restaurant_ratings_sorted)
    '''

    '''
    plt.scatter(restaurant_ratings_sorted['rating'], restaurant_ratings_sorted['visit'])
    plt.title('Number of visit per restaurant rating mean')
    plt.ylabel('Number of visits')
    plt.xlabel('Restaurant rating mean')
    plt.show()
    '''

    '''
    #CREAR K-MEAN DE NOMBRE DE VISITES DEL RESTAURANT RESPECTE LA MITJANA DE PUNTUACIONS.
    X = restaurant_ratings_sorted.to_numpy()

    km = KMeans(
        n_clusters=3, init='random',
        n_init=10, max_iter=300, random_state=0
    )
    y_km = km.fit_predict(X)

    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    '''

    for i in range(10):
        current_group = user_group_mean[user_group_mean['group'] != i].index
        user_group_mean_9 = user_group_mean.drop(current_group)
        user_group_mean_9.set_index('userID', inplace=True)

        X = user_group_mean_9.to_numpy()
        size = 100
        km = KMeans(
            n_clusters=size, init='random',
            n_init=10, max_iter=300, random_state=0
        )
        y_km = km.fit_predict(X)
        user_group_mean_9['k-means'] = y_km
        user_group_mean_9.to_excel(r'C:\\Users\\Sancho\\Desktop\\kmeans2\\group' + str(i) + '.xlsx')

    '''
    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )
    plt.scatter(
        X[y_km == 3, 0], X[y_km == 3, 1],
        s=50, c='red',
        marker='v', edgecolor='black',
        label='cluster 4'
    )
    plt.scatter(
        X[y_km == 4, 0], X[y_km == 4, 1],
        s=50, c='yellow',
        marker='v', edgecolor='black',
        label='cluster 5'
    )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    '''
except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
