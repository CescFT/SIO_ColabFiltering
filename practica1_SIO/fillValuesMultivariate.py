import psycopg2
import pandas as pd
import openpyxl
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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

    pivot_ratings = all_ratings.pivot_table(index=['userID'], columns=['restaurantID'], values='rating')
    print(pivot_ratings)

    # Aquest for es per cada grup de clusterin, del 0 al 9.
    count = 0
    all_users_predicted = pd.DataFrame(columns=['userID', 'restaurantID'])
    for count in range(10):
        # Legim les dades de cadascun dels excels.
        group_data = pd.read_excel("..\\kmeans2\\group" + str(count) + ".xlsx")

        # Per cada excel iterem el usuaries que pertanyen a cada grup de kmeans del 0 al 99.
        for i in range(100):
            # Agafem tota l'informació de cada kmeans.
            all_group_data = group_data.loc[group_data['k-means'] == i]

            # Ens quedem amb els IDs dels usuaris que pertanyen al kmeans.
            user_group_id = pd.DataFrame(columns=['userID'])
            user_group_id['userID'] = all_group_data['userID']
            user_group_id.reset_index(drop=True, inplace=True)

            # final_df serà la variable qua guardarà el conjunt de dades.
            final_df = pd.DataFrame(columns=['restaurantID', 'rating'])

            # Iterem tots els usuaris del grup I.
            for num_user in range(user_group_id.size):
                print(str(num_user) + ':' + str(user_group_id.size))
                # Aqui agafem els restaurants i les putuacions de l'usuari que volem omplir les dades buides.
                rating_user = pivot_ratings.loc[[user_group_id['userID'].values[num_user]], :].stack(
                    dropna=False).to_numpy()
                pred_user = pd.DataFrame(columns=['restaurantID', 'rating'])
                pred_user['restaurantID'] = np.arange(1, 101, dtype=np.int64)
                pred_user['rating'] = rating_user

                # Comprovem que l'usuari té puntuacions buides.
                if pred_user['rating'].isna().sum() != 0:
                    # Agafem tots els altres usuaris i els guardem al conjunt de dades.
                    for x in np.append(np.arange(0, num_user), np.arange(num_user + 1, user_group_id.size)):
                        rating_user = pivot_ratings.loc[[user_group_id['userID'].values[x]], :].stack(
                            dropna=False).to_numpy()
                        aux_df = pd.DataFrame(columns=['restaurantID', 'rating'])
                        aux_df['restaurantID'] = np.arange(1, 101, dtype=np.int64)
                        aux_df['rating'] = rating_user
                        final_df = pd.concat([final_df, aux_df])

                    # Omplim les dades de l'usuari que hem trobat.
                    final_series = final_df.to_numpy()
                    imp = IterativeImputer(max_iter=10, random_state=0)
                    imp.fit(final_series)
                    X_test = pred_user.to_numpy()
                    user_predicted = pd.DataFrame(data=imp.transform(X_test), columns=['restaurantID', 'rating'])
                    user_predicted['userID'] = 'User' + str(user_group_id['userID'].values[num_user])
                    user_predicted['restaurantID'] = 'Restaurant' + pred_user['restaurantID'].astype(str)
                    all_users_predicted = pd.concat([all_users_predicted, user_predicted])

        break
    all_users_predicted.to_csv('C:\\Users\\Sancho\\Desktop\\predicted.csv', index=False)

    '''
    for i in range(10):
        current_group = user_group_mean[user_group_mean['group'] != i].index
        user_group_mean_9 = user_group_mean.drop(current_group)
        user_group_mean_9.set_index('userID', inplace=True)

        X = user_group_mean_9.to_numpy()
        size = int(user_group_mean_9['group'].size)
        size = int(size / 2)
        km = KMeans(
            n_clusters=size, init='random',
            n_init=10, max_iter=300, random_state=0
        )
        y_km = km.fit_predict(X)
        user_group_mean_9['k-means'] = y_km
        user_group_mean_9.to_excel(r'C:\\Users\\Sancho\\Desktop\\group' + str(i) + '.xlsx')
    '''

except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
