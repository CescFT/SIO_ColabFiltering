import psycopg2
import pandas as pd
import openpyxl
import numpy as np
from sklearn.impute import KNNImputer

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

    user_pos = user_ratings[user_ratings['rating'] > 0]
    user_pos = user_pos.sort_values('rating')
    user_pos = user_pos.reset_index(drop=True)
    print(user_pos)

    user_neg = user_ratings[user_ratings['rating'] < 0]
    user_neg = user_neg.sort_values('rating')
    user_neg = user_neg.reset_index(drop=True)

    pivot_ratings = all_ratings.pivot_table(index=['userID'], columns=['restaurantID'], values='rating')
    print(pivot_ratings)

    all_users_predicted = pd.DataFrame(columns=['userID', 'restaurantID'])

    template_df_concat = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])

    # final_df serà la variable qua guardarà el conjunt de dades.
    final_df = pd.DataFrame(columns=['restaurantID', 'rating'])

    # Guardem estructura reduïda de la sortida.
    template_df_nan = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])

    x = 0
    lon = 0
    # Per cada excel iterem el usuaries que pertanyen a cada grup de kmeans del 0 al 99.
    for i in range(user_pos['userID'].size):
        print(str(i) + ':' + str(user_pos['userID'].size))
        user_id = int(user_pos.loc[i]['userID'])

        # Iterem tots els usuaris del grup I.
        template_df = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])

        # Aqui agafem els restaurants i les putuacions de l'usuari que volem omplir les dades buides.
        rating_user = pivot_ratings.loc[user_id, :]
        rating_user.reset_index(inplace=True, drop=True)

        info_user = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])
        info_user['restaurantID'] = np.arange(1, 101, dtype=np.int64)
        info_user['rating'] = rating_user
        info_user['userID'] = 'User' + str(user_id)

        template_df_concat = template_df_concat.append(info_user, ignore_index=True)

        final_df = final_df.append(info_user[['restaurantID', 'rating']], ignore_index=True)

        if i % 10 == 0 and i != 0 and i + 10 < user_pos['userID'].size - 1:
            template_df_concat['isnan'] = template_df_concat['rating'].apply(lambda x: np.isnan(x))
            template_df_concat.reset_index(drop=True, inplace=True)
            lon = i - x
            x = i
            inputer = KNNImputer(n_neighbors=lon + 1, weights='distance')
            inputer_output = inputer.fit_transform(final_df)[:, 1]
            template_df_concat['rating'] = pd.Series(inputer_output)
            template_df_nan = template_df_nan.append(template_df_concat[template_df_concat['isnan'] == True],
                                                     ignore_index=True)

        elif i == user_pos['userID'].size - 1:
            template_df_concat['isnan'] = template_df_concat['rating'].apply(lambda x: np.isnan(x))
            template_df_concat.reset_index(drop=True, inplace=True)
            lon = i - x
            x = i
            inputer = KNNImputer(n_neighbors=lon + 1, weights='distance')
            inputer_output = inputer.fit_transform(final_df)[:, 1]
            template_df_concat['rating'] = pd.Series(inputer_output)
            template_df_nan = template_df_nan.append(template_df_concat[template_df_concat['isnan'] == True],
                                                     ignore_index=True)

    template_df_nan.reset_index(inplace=True, drop=True)
    template_df_nan['restaurantID'] = 'Restaurant' + template_df_concat['restaurantID'].astype(str)
    template_df_nan.drop('isnan', axis=1, inplace=True)
    template_df_nan.reset_index(inplace=True, drop=True)
    template_df_nan.to_csv('C:\\Users\\Sancho\\Desktop\\predicted_negative.csv', index=False)

except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
