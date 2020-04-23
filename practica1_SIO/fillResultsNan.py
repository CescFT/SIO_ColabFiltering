import random
from ast import literal_eval
import pandas as pd
import numpy as np

try:

    csv_pos = pd.read_csv('..\\..\\predicted_positive.csv')
    csv_pos['rating'] = csv_pos['rating'].apply(lambda x: round(x, 4))
    csv_neg = pd.read_csv('..\\..\\predicted_negative.csv')
    csv_neg['rating'] = csv_neg['rating'].apply(lambda x: round(x, 4))
    csv_final = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])
    csv_final = pd.concat([csv_pos, csv_neg])
    print(csv_final)
    template_csv = pd.read_csv('..\\..\\PlantillaPrediccions.csv', delimiter=';',
                               names=['userID', 'restaurantID', 'rating'])
    template_copy = template_csv.copy()
    template_copy.drop('rating', axis=1, inplace=True)


    final_df = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])
    list_row = []
    for i in range(template_copy['userID'].size):
        print(str(i)+':'+str(template_copy['userID'].size))
        array = np.asarray(template_copy.iloc[i])
        row_to_fill = csv_final[csv_final.userID.isin([array[0]]) & csv_final.restaurantID.isin([array[1]])]
        #row_to_fill = csv_final.loc[(csv_final['userID'] == array[0]) & (csv_final['restaurantID'] == array[1])]
        dict_row = pd.DataFrame.to_dict(row_to_fill)
        list_row.append(dict_row)
        #final_df = pd.concat([final_df, row_to_fill], ignore_index=True)
        if i+1 % 100:
            final_df = pd.DataFrame.from_dict(list_row)
            final_df.to_csv('..\\tokenName.csv', mode='a', index=False, header=False, sep=';')
            list_row = []




except Exception as e:
    print(e)
