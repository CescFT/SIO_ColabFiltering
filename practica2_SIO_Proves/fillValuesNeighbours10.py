import psycopg2
import pandas as pd
import openpyxl
import numpy as np
from sklearn.impute import KNNImputer


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

    all_users_predicted = pd.DataFrame(columns=['userID', 'restaurantID'])
    for count in range(10):
        # Legim les dades de cadascun dels excels.
        group_data = pd.read_excel("..\\kmeans2\\group" + str(count) + ".xlsx")

        # Per cada excel iterem el usuaries que pertanyen a cada grup de kmeans del 0 al 99.
        for i in range(100):
            print(str(count) + '-' + str(i) + ':' + str(100))
            # Agafem tota l'informació de cada kmeans.
            all_group_data = group_data.loc[group_data['k-means'] == i]

            # Ens quedem amb els IDs dels usuaris que pertanyen al kmeans.
            user_group_id = pd.DataFrame(columns=['userID'])
            user_group_id['userID'] = all_group_data['userID']
            user_group_id.reset_index(drop=True, inplace=True)

            # final_df serà la variable qua guardarà el conjunt de dades.
            final_df = pd.DataFrame(columns=['restaurantID', 'rating'])

            # Iterem tots els usuaris del grup I.
            all_users_ratings = pd.DataFrame(columns=['restaurantID', 'rating'])
            template_df = pd.DataFrame(columns=['userID', 'restaurantID'])
            template_df_concat = pd.DataFrame(columns=['userID', 'restaurantID'])
            temporal_user = pd.DataFrame(columns=['userID', 'restaurantID'])
            temporal_rating = np.ndarray(0)
            for num_user in range(user_group_id.size):
                # Aqui agafem els restaurants i les putuacions de l'usuari que volem omplir les dades buides.
                rating_user = pivot_ratings.loc[[user_group_id['userID'].values[num_user]], :].stack(
                    dropna=False).to_numpy()
                pred_user = pd.DataFrame(columns=['restaurantID', 'rating'])
                pred_user['restaurantID'] = np.arange(1, 101, dtype=np.int64)
                pred_user['rating'] = rating_user
                template_df['restaurantID'] = 'Restaurant' + pred_user['restaurantID'].astype(str)
                template_df['userID'] = 'User' + user_group_id['userID'].values[num_user].astype(str)
                temporal_user = pd.concat([temporal_user, template_df])

                template_df_concat = pd.concat([template_df_concat, template_df])
                all_users_ratings = pd.concat([all_users_ratings, pred_user])

                if num_user % 10 == 0 and num_user != 0 and num_user + 10 < user_group_id['userID'].size - 1:
                    inputer = KNNImputer(n_neighbors=all_users_ratings.size, weights='distance')
                    inputer_output = inputer.fit_transform(all_users_ratings)[:, 1]
                    temporal_user.reset_index(drop=True, inplace=True)
                    temporal_user['rating'] = pd.Series(inputer_output)
                    temporal_rating = np.concatenate([temporal_rating, inputer_output])

                    template_df = pd.DataFrame(columns=['userID', 'restaurantID'])
                    all_users_ratings = pd.DataFrame(columns=['restaurantID', 'rating'])

                elif num_user == user_group_id['userID'].size - 1:
                    inputer = KNNImputer(n_neighbors=all_users_ratings.size, weights='distance')
                    inputer_output = inputer.fit_transform(all_users_ratings)[:, 1]
                    temporal_user.reset_index(drop=True, inplace=True)
                    temporal_user['rating'] = pd.Series(inputer_output)
                    temporal_rating = np.concatenate([temporal_rating, inputer_output])

                    template_df = pd.DataFrame(columns=['userID', 'restaurantID'])
                    all_users_ratings = pd.DataFrame(columns=['restaurantID', 'rating'])

            template_df_concat.reset_index(drop=True, inplace=True)
            template_df_concat['rating'] = pd.Series(temporal_rating)
            all_users_predicted = pd.concat([all_users_predicted, template_df_concat])

    all_users_predicted.to_csv('C:\\Users\\Sancho\\Desktop\\predicted_neighbours10.csv', index=False)

except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
