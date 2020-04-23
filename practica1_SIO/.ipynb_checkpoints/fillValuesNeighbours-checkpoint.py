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
    
    user_neg = user_ratings[user_ratings['rating'] < 0]
    user_neg = user_neg.sort_values('rating')
    user_neg = user_neg.reset_index(drop=True)
    
    pivot_ratings = all_ratings.pivot_table(index=['userID'], columns=['restaurantID'], values='rating')
    print(pivot_ratings)

    all_users_predicted = pd.DataFrame(columns=['userID', 'restaurantID'])
    
            # Legim les dades de cadascun dels excels.
            group_data = pd.read_excel("..\\kmeans2\\group" + str(count) + ".xlsx")

            # Per cada excel iterem el usuaries que pertanyen a cada grup de kmeans del 0 al 99.
            for i in range(100):
                print(str(count)+'-'+str(i)+':'+str(100))
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
                for num_user in range(user_group_id.size):
                    # Aqui agafem els restaurants i les putuacions de l'usuari que volem omplir les dades buides.
                    rating_user = pivot_ratings.loc[[user_group_id['userID'].values[num_user]], :].stack(
                        dropna=False).to_numpy()
                    pred_user = pd.DataFrame(columns=['restaurantID', 'rating'])
                    pred_user['restaurantID'] = np.arange(1, 101, dtype=np.int64)
                    pred_user['rating'] = rating_user
                    template_df['restaurantID'] = 'Restaurant' + pred_user['restaurantID'].astype(str)
                    template_df['userID'] = 'User' + user_group_id['userID'].values[num_user].astype(str)
                    template_df_concat = pd.concat([template_df_concat, template_df])
                    all_users_ratings = pd.concat([all_users_ratings, pred_user])
                inputer = KNNImputer(n_neighbors=all_users_ratings.size, weights='distance')
                inputer_output = inputer.fit_transform(all_users_ratings)[:, 1]
                template_df_concat.reset_index(drop=True, inplace=True)
                template_df_concat['rating'] = pd.Series(inputer_output)
                all_users_predicted = pd.concat([all_users_predicted, template_df_concat])

        all_users_predicted.to_csv('C:\\Users\\Sancho\\Desktop\\predicted_neighbours.csv', index=False)

except Exception as e:
    print(e)

finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
