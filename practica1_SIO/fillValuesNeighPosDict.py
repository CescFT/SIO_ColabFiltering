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

    all_users_predicted = []
    # all_users_predicted = pd.DataFrame(columns=['userID', 'restaurantID'])

    template_df_concat_dict = []
    # template_df_concat = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])

    # final_df serà la variable qua guardarà el conjunt de dades.
    final_df_dict = []
    # final_df = pd.DataFrame(columns=['restaurantID', 'rating'])

    # Guardem estructura reduïda de la sortida.
    template_df_nan = []
    # template_df_nan = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])

    user_10_dict = []
    ratings_users = []

    x = 0
    lon = 0
    # Per cada excel iterem el usuaries que pertanyen a cada grup de kmeans del 0 al 99.
    for i in range(user_pos['userID'].size):
        print(str(i) + ':' + str(user_pos['userID'].size))
        user_id = int(user_pos.loc[i]['userID'])

        # Iterem tots els usuaris del grup I.
        template_df = []
        # template_df = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])

        # Aqui agafem els restaurants i les putuacions de l'usuari que volem omplir les dades buides.
        rating_user = pivot_ratings.loc[user_id, :]
        rating_user.reset_index(inplace=True, drop=True)

        for count in range(100):
            pred_user = {'userID': 'User' + str(user_id), 'restaurantID': count + 1,
                         'rating': rating_user[count]}
            template_df_concat_dict.append(pred_user)
            user_10_dict.append(pred_user)
        # pred_user = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])

        # template_df_concat = pd.concat([template_df_concat, pred_user])
        # final_df = pd.concat([final_df, pred_user[['restaurantID', 'rating']]])

        if i % 10 == 0 and i != 0 and i + 10 < user_pos.size - 1:
            template_df_concat = pd.DataFrame.from_dict(template_df_concat_dict)
            template_df_concat['isnan'] = template_df_concat['rating'].apply(lambda x: np.isnan(x))
            template_df_concat.reset_index(drop=True, inplace=True)
            lon = i - x
            x = i
            inputer = KNNImputer(n_neighbors=lon + 1, weights='distance')
            user_10_dict = pd.DataFrame.from_dict(user_10_dict)
            user_10_dict = user_10_dict.drop('userID', axis=1)
            inputer_output = inputer.fit_transform(user_10_dict)[:, 1]
            for j in pd.Series(inputer_output):
                rating = {'rating': j}
                ratings_users.append(rating)
            template_df_concat['rating'] = pd.DataFrame.from_dict(ratings_users)
            template_df_concat = template_df_concat[template_df_concat['isnan'] == True]
            aux_dict = template_df_concat.to_dict('records')
            # template_df_nan = pd.concat([template_df_nan, template_df_concat[template_df_concat['isnan'] == True]])
            user_10_dict = []

            if i % 1000 == 0:
                for y in aux_dict:
                    template_df_nan.append(y)
                template_df_nan = pd.DataFrame.from_dict(template_df_nan)
                template_df_nan['restaurantID'] = 'Restaurant' + template_df_nan['restaurantID'].astype(str)
                template_df_nan.drop('isnan', axis=1, inplace=True)
                template_df_nan.to_csv('C:\\Users\\Sancho\\Desktop\\predicted_positive.csv', mode='a', index=False,
                                       header=False)
                template_df_nan = []
                template_df_concat_dict = []

        elif i == user_pos.size - 1:
            template_df_concat = pd.DataFrame.from_dict(template_df_concat_dict)
            template_df_concat['isnan'] = template_df_concat['rating'].apply(lambda x: np.isnan(x))
            template_df_concat.reset_index(drop=True, inplace=True)
            lon = i - x
            x = i
            inputer = KNNImputer(n_neighbors=lon + 1, weights='distance')
            user_10_dict = pd.DataFrame.from_dict(user_10_dict)
            user_10_dict = user_10_dict.drop('userID', axis=1)
            inputer_output = inputer.fit_transform(user_10_dict)[:, 1]
            for j in pd.Series(inputer_output):
                rating = {'rating': j}
                ratings_users.append(rating)
            template_df_concat['rating'] = pd.DataFrame.from_dict(ratings_users)
            template_df_concat = template_df_concat[template_df_concat['isnan'] == True]
            aux_dict = template_df_concat.to_dict('records')
            # template_df_nan = pd.concat([template_df_nan, template_df_concat[template_df_concat['isnan'] == True]])
            user_10_dict = []

    for y in aux_dict:
        template_df_nan.append(y)
    template_df_nan = pd.DataFrame.from_dict(template_df_nan)
    template_df_nan['restaurantID'] = 'Restaurant' + template_df_nan['restaurantID'].astype(str)
    template_df_nan.drop('isnan', axis=1, inplace=True)
    template_df_nan.to_csv('C:\\Users\\Sancho\\Desktop\\predicted_positive.csv', mode='a', index=False, header=False)

except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
