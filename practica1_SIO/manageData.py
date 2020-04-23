import psycopg2
import pandas as pd
import numpy as np

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
    print(user_ratings)

    restaurant_ratings = all_ratings.drop(['userID'], axis=1)
    restaurant_ratings = restaurant_ratings.groupby('restaurantID', as_index=False).mean()
    print(restaurant_ratings)

    restaurant_visits = all_ratings.drop(['rating'], axis=1)
    restaurant_visits = restaurant_visits.groupby('restaurantID', as_index=False).count().rename(
        columns={'userID': 'visit'})
    print(restaurant_visits)

    user_visits = all_ratings.drop(['rating'], axis=1)
    user_visits = user_visits.groupby('userID', as_index=False).count().rename(columns={'restaurantID': 'visit'})
    print(user_visits)

    print(user_visits.groupby(user_visits['visit']==100).count())

    '''
    pivot_ratings = all_ratings.pivot_table(index=['userID'], columns=['restaurantID'], values='rating')
    print(pivot_ratings)
    
    user_sd = pd.DataFrame(columns=['std'])
    user_sd['std'] = pivot_ratings.std(axis=1, skipna=True)
    user_sd = user_sd.reset_index()
    print(user_sd)

    restaurant_sd = pd.DataFrame(columns=['std'])
    restaurant_sd['std'] = pivot_ratings.std(skipna=True)
    restaurant_sd = restaurant_sd.reset_index()
    print(restaurant_sd)

    prova_df = user_visits
    prova_df['rating'] = user_ratings['rating']
    print(user_ratings)
    print(prova_df)

    plt.scatter(prova_df['rating'], prova_df['visit'])
    plt.title('Number of visit per restaurant rating mean')
    plt.ylabel('Number of visits')
    plt.xlabel('Restaurant rating mean')
    plt.show()
    '''

    '''
    interval_df = all_ratings.drop(['restaurantID'], axis=1)
    interval_df['rating'] = interval_df['rating'].round(0)
    interval_df['rating'].replace({10:9}, inplace=True)
    interval_df = interval_df.groupby('rating', as_index=False).count().rename(columns={'userID':'count'})
    print(interval_df)

    count_plot = interval_df['count']
    ax = count_plot.plot(kind='bar')
    ax.set_xticklabels(np.arange(-10, 10))
    ax.set_title('Number of occurrences per interval of ratings')
    ax.set_ylabel('Number of occurrences')
    ax.set_xlabel('Interval of ratings')

    plt.show()
    '''

    '''
    delete_decimals = all_ratings
    delete_decimals['rating'] = all_ratings['rating'].round(0)
    print(all_ratings)
    print(delete_decimals)
    delete_decimals = delete_decimals.pivot_table(index=['userID'], columns=['restaurantID'], values='rating')
    user_mode = delete_decimals.mode(axis=1)[0]
    print(user_mode)
    restaurant_mode = delete_decimals.mode()
    print(restaurant_mode)
    '''

    '''
    restaurant_ratings_sorted = pd.DataFrame(columns=['rating', 'visit'])
    restaurant_ratings_sorted['rating'] = restaurant_ratings['rating']
    restaurant_ratings_sorted['visit'] = restaurant_visits['visit']
    restaurant_ratings_sorted = restaurant_ratings_sorted.sort_values(by=['rating'])
    print(restaurant_ratings_sorted)
   
    plt.scatter(restaurant_ratings_sorted['rating'], restaurant_ratings_sorted['visit'])
    plt.title('Number of visit per restaurant rating mean')
    plt.ylabel('Number of visits')
    plt.xlabel('Restaurant rating mean')
    plt.show()
    '''

    '''
    restaurant_ratings_sorted['rating'] = restaurant_ratings_sorted['rating'].round(0)
    restaurant_ratings_sorted = restaurant_ratings_sorted.groupby(['rating']).mean().reset_index()
    print(restaurant_ratings_sorted)

    plt.bar(restaurant_ratings_sorted['rating'], restaurant_ratings_sorted['visit'])
    plt.title('Number of visit per interval restaurant rating mean')
    plt.ylabel('Number of visits')
    plt.xlabel('Restaurant rating mean')
    plt.show()
    '''

    '''
    mode_group_user = pd.DataFrame(columns=['mode', 'mean'])
    mode_group_user['mode'] = user_mode
    mode_group_user['mean'] = user_ratings['rating']
    mode_group_user = mode_group_user.groupby(['mode']).mean().reset_index()
    print(mode_group_user)
    
    plt.scatter(mode_group_user['mode'], mode_group_user['mean'])
    plt.title('Mean rating per interval mode') 
    plt.ylabel('Mean rating')
    plt.xlabel('Interval mode')
    z = np.polyfit(mode_group_user['mode'], mode_group_user['mean'], 1)
    y = np.poly1d(z)
    plt.plot(mode_group_user['mode'], y(mode_group_user['mode']), "r--")
    plt.show()
    '''

    '''
    mean_group_user = pd.DataFrame(columns=['mode', 'mean'])
    mean_group_user['mode'] = user_mode
    mean_group_user['mean'] = user_ratings['rating']
    mean_group_user['mean'] = user_ratings['rating'].round(0)
    mean_group_user = mean_group_user.groupby(['mean']).mean().reset_index()
    print(mean_group_user)

    plt.scatter(mean_group_user['mean'], mean_group_user['mode'])
    plt.title('Mode rating per interval mean')
    plt.ylabel('Mode rating')
    plt.xlabel('Interval mean')
    z = np.polyfit(mean_group_user['mean'], mean_group_user['mode'], 1)
    y = np.poly1d(z)
    plt.plot(mean_group_user['mean'], y(mean_group_user['mean']), "r--")
    plt.show()
    '''

    '''
    mean_counts_user = pd.DataFrame(columns=['count'])
    mean_counts_user['count'] = user_ratings['rating'].round(0)
    mean_counts_user = mean_counts_user['count'].value_counts().reset_index()
    mean_counts_user = mean_counts_user.rename(columns={'index':'mean'})
    mean_counts_user = mean_counts_user.sort_values(by=['mean'])
    print(mean_counts_user)
    
    plt.scatter(mean_counts_user['mean'], mean_counts_user['count'])
    plt.title('Number of users interval mean occurrences')
    plt.ylabel('Number of occurrences')
    plt.xlabel('Interval of mean')
    plt.show()
    '''

    '''
    mean_counts_restaurant = pd.DataFrame(columns=['count'])
    mean_counts_restaurant['count'] = restaurant_ratings['rating'].round(0)
    mean_counts_restaurant = mean_counts_restaurant['count'].value_counts().reset_index()
    mean_counts_restaurant = mean_counts_restaurant.rename(columns={'index': 'mean'})
    mean_counts_restaurant = mean_counts_restaurant.sort_values(by=['mean'])
    print(mean_counts_restaurant)

    plt.scatter(mean_counts_restaurant['mean'], mean_counts_restaurant['count'])
    plt.title('Number of restaurants interval mean occurrences')
    plt.ylabel('Number of occurrences')
    plt.xlabel('Interval of mean')
    plt.show()
    '''

    '''
    prob_df = pd.DataFrame(columns=['count'])
    prob_df['count'] = all_ratings['rating'].round(1)
    print(all_ratings)
    print(prob_df)
    prob_df = prob_df['count'].value_counts().reset_index()
    prob_df['probability'] = prob_df['count'] / len(user_ratings)
    prob_df = prob_df.rename(columns={'index': 'rating'})
    prob_df = prob_df.sort_values(by=['rating'])
    prob_df = prob_df.reset_index(drop=True)
    print(prob_df)
    
    plt.scatter(prob_df['rating'], prob_df['probability'])
    plt.title('Probability of each rating')
    plt.ylabel('Probability')
    plt.xlabel('Rating')
    plt.show()
    '''

    '''
    std_mean_user = pd.DataFrame(user_sd['std'], columns=['std']).reset_index(drop=True)
    std_mean_user['mean'] = user_ratings['rating']
    print(std_mean_user)

    plt.scatter(std_mean_user['mean'], std_mean_user['std'])
    plt.title('Population deviation per user rating mean')
    plt.ylabel('Population deviation')
    plt.xlabel('User rating mean')
    plt.show()
    '''

    '''
    std_mean_restaurant = pd.DataFrame(restaurant_sd['std'], columns=['std']).reset_index(drop=True)
    std_mean_restaurant['mean'] = restaurant_ratings['rating']
    print(std_mean_restaurant)

    plt.scatter(std_mean_restaurant['mean'], std_mean_restaurant['std'])
    plt.title('Population deviation per restaurant rating mean')
    plt.ylabel('Population deviation')
    plt.xlabel('Restaurant rating mean')
    plt.show()
    '''

    '''
    std_mode_restaurant = pd.DataFrame(user_sd['std'], columns=['std']).reset_index(drop=True)
    std_mode_restaurant['mode'] = user_mode.reset_index(drop=True)
    print(std_mode_restaurant)

    plt.scatter(std_mode_restaurant['mode'], std_mode_restaurant['std'])
    plt.title('Population deviation per user rating mode')
    plt.ylabel('Population deviation')
    plt.xlabel('User rating mode')
    plt.show()
    '''

    '''
    user_boolean = user_ratings
    user_boolean['positive'] = user_boolean['rating'] >= 0
    user_boolean = user_boolean.groupby('positive', as_index=False).count()
    print(user_boolean)

    
    plt.bar(user_boolean['positive'], user_boolean['userID'])
    plt.title('Percentage of positive and negative users')
    plt.ylabel('Number of users')
    plt.xlabel('User positivism')
    plt.xticks(user_boolean['positive'], user_boolean['positive'])
    for a, b in zip(user_boolean['positive'], user_boolean['userID']):
        plt.text(a, b, str(round(b / len(user_ratings) * 100, 1)), ha='center')
    plt.show()
    '''

    '''
    user_boolean = all_ratings
    user_ratings_intervals = pd.DataFrame(columns=['range', 'count'])
    user_ratings_intervals['range'] = np.arange(-10, 9, 2)
    j = np.arange(-8, 11, 2)
    counters = np.array([0])
    x = -10
    count = 0
    print(all_ratings)
    for i in j:
        if i == 10:
            i = 11
        user_boolean['positive'] = (all_ratings['rating'] >= x) & (all_ratings['rating'] < i)
        x = i
        user_boolean = user_boolean.groupby('positive', as_index=False).count()
        user_boolean = user_boolean.drop([0])
        counters = np.append(counters, user_boolean['rating'])
        user_boolean = all_ratings
        count = count + 1
    counters = np.delete(counters, 0, axis=0)
    user_ratings_intervals['count'] = counters
    print(user_ratings_intervals)

    plt.bar(user_ratings_intervals['range'], user_ratings_intervals['count'])
    plt.title('User rating per group by intervals')
    plt.ylabel('Number of ratings')
    plt.xlabel('Rating intervals')
    plt.xticks(user_ratings_intervals['range'], user_ratings_intervals['range'])
    for a, b in zip(user_ratings_intervals['range'], user_ratings_intervals['count']):
        plt.text(a, b, str(round(b / len(user_ratings) * 100, 1)), ha='center')
    plt.show()
    '''

    '''
    restaurant_boolean = restaurant_ratings
    restaurant_boolean['positive'] = restaurant_boolean['rating'] >= 0
    restaurant_boolean = restaurant_boolean.groupby('positive', as_index=False).count()
    print(restaurant_boolean)

    plt.bar(restaurant_boolean['positive'], restaurant_boolean['restaurantID'])
    plt.title('Percentage of positive and negative restaurants')
    plt.ylabel('Number of restaurants')
    plt.xlabel('Restaurant positivism')
    plt.xticks(restaurant_boolean['positive'], restaurant_boolean['positive'])
    for a, b in zip(restaurant_boolean['positive'], restaurant_boolean['restaurantID']):
        plt.text(a, b, str(round(b / len(restaurant_ratings) * 100, 1)), ha='center')
    plt.show()
    '''

    '''
    different_users_same_rating = pd.DataFrame(all_ratings.drop(['restaurantID'], axis=1), columns=['userID', 'rating'])
    different_users_same_rating['rating'] = different_users_same_rating['rating'].round(1)
    different_users_same_rating = different_users_same_rating.groupby(['rating', 'userID']).count().reset_index()
    different_users_same_rating = different_users_same_rating.groupby(['rating']).count().reset_index()
    print(different_users_same_rating)
    
    plt.scatter(different_users_same_rating['userID'], prob_df['probability'])
    plt.title('Probability per different users with same rating')
    plt.ylabel('Probability')
    plt.xlabel('Number of users')
    z = np.polyfit(different_users_same_rating['userID'], prob_df['probability'], 1)
    y = np.poly1d(z)
    plt.plot(different_users_same_rating['userID'], y(different_users_same_rating['userID']), "r--")
    plt.show()
    '''

    '''
    PROVA DE RESTAURANTS PER INTERVALS (NO SE VEURE RES)
    ratings_interval_user = all_ratings.drop(['userID'], axis=1)
    ratings_interval_user['range'] = pd.cut(ratings_interval_user['rating'], np.arange(-10, 11, 5))
    ratings_interval_user = ratings_interval_user.groupby(['restaurantID', 'range']).count().reset_index()
    ratings_interval_user = ratings_interval_user.dropna().reset_index(drop=True)
    ratings_interval_user = ratings_interval_user.reset_index()
    ratings_interval_user = ratings_interval_user.drop(ratings_interval_user[ratings_interval_user.restaurantID > 25].index)
    print(ratings_interval_user)

    plt.bar(ratings_interval_user['index'], ratings_interval_user['rating'])
    plt.title('Restaurants ratings per number of repeated')
    plt.ylabel('Number of repeated')
    plt.xlabel('Restaurants ratings intervals')
    for a, b in zip(ratings_interval_user['index'], ratings_interval_user['rating']):
        plt.text(a, b, str(ratings_interval_user.at[a, 'restaurantID']), ha='center')
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
