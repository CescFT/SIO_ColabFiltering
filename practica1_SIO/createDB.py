import psycopg2
import pandas as pd
import numpy as np

def create_clients_table(data):
    # Create a Data Frame with the user name and an identifier starts with 1.
    client_df = pd.DataFrame()
    #client_df = pd.DataFrame(data, columns=['USERS', 'id'])
    #client_df['USERS'] = data['DATASET']
    client_df = pd.DataFrame(data, columns=['id'])
    client_df['id'] = np.arange(1, len(client_df) + 1)
    return client_df

def create_restaurant_table(data):
    # Create a Data Frame with the restaurant name and an identifier starts with 1.
    restaurant_df = pd.DataFrame()
    restaurant_names = pd.DataFrame(data.iloc[0]).iloc[1:]
    restaurant_df = pd.DataFrame(restaurant_names, columns=['id']).reset_index(drop=True)#.rename_axis('RESTAURANT').reset_index()
    restaurant_df['id'] = np.arange(1, len(restaurant_names) + 1)
    return restaurant_df

def create_relations_table(data, client_df, restaurant_df):
    try:
        # Create a Data Frame with each user repeated for all the restaurants.
        relation_df = pd.DataFrame()
        client_id = pd.DataFrame(np.repeat(client_df['id'].values, 100, axis=0))
        client_id = pd.DataFrame(client_id.values, columns=['username'])

        # Create a Data Frame with all the restaurants repeated, to make the relation with 1 user and all the restaurants, and this for all users.
        restaurant_id = pd.DataFrame(restaurant_df['id']).rename(columns={"id": "restaurant"})
        restaurant_id = pd.concat([restaurant_id] * len(client_df)).reset_index(drop=True)

        relation_df = pd.concat([client_id, restaurant_id], axis=1, sort=False)

        ratings_df = pd.DataFrame(data).T
        ratings_df = ratings_df.drop('DATASET').T
        ratings_df = pd.DataFrame(ratings_df.values.ravel(), columns=['rating'])

        relation_df = pd.concat([relation_df, ratings_df], axis=1, sort=False)

        relation_df = relation_df[relation_df.rating != 99].reset_index().drop(['index'], axis=1)

        return relation_df

    except Exception as error:
        return error

try:
    connection = psycopg2.connect(user="postgres",
                                  password="aleixDatabase",
                                  host="localhost",
                                  port="3306",
                                  database="siodb")

    cursor = connection.cursor()

    cursor.execute('''DROP TABLE users, restaurants, ratings''')

    create_table_users = '''CREATE TABLE users (
                                id BIGINT NOT NULL UNIQUE,
                                PRIMARY KEY (id)
                                );'''

    create_table_restaurants = '''CREATE TABLE restaurants (
                                id BIGINT NOT NULL UNIQUE,
                                PRIMARY KEY (id)
                                );'''

    create_table_ratings = '''CREATE TABLE ratings (
                                username BIGINT NOT NULL,
                                restaurant BIGINT NOT NULL,
                                rating REAL NOT NULL,
                                FOREIGN KEY (username) REFERENCES users (id),
                                FOREIGN KEY (restaurant) REFERENCES restaurants (id)
                                );'''

    data = pd.read_csv("../../dataset.csv", delimiter=";")
    clients_df = create_clients_table(data)
    restaurants_df = create_restaurant_table(data)
    relations_df = create_relations_table(data, clients_df, restaurants_df)

    insert_users_id = '''INSERT INTO users (id) VALUES (%s)'''
    insert_restaurants_id = '''INSERT INTO restaurants (id) VALUES (%s)'''
    insert_relations = '''INSERT INTO ratings (username, restaurant, rating) VALUES (%s, %s, %s)'''

    # Now call all queries.
    cursor.execute(create_table_users)
    cursor.execute(create_table_restaurants)
    cursor.execute(create_table_ratings)
    cursor.executemany(insert_users_id, clients_df.values.tolist())
    cursor.executemany(insert_restaurants_id, restaurants_df.values.tolist())
    cursor.executemany(insert_relations, relations_df.values.tolist())

    connection.commit()
    print("Table created successfully in PostgreSQL ")

except (Exception, psycopg2.DatabaseError) as error:
    print("Error while creating PostgreSQL table", error)
finally:
    # closing database connection.
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
