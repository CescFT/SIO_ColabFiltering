import pandas as pd
import numpy as np
from ast import literal_eval

try:
    csv_pos = pd.read_csv('..\\tokenName.csv', sep=';')
    csv_pos2 = pd.read_csv('..\\tokenName2.csv', sep=';')
    aux_df = csv_pos[csv_pos['userID'] == '{}']

    ind = aux_df.index
    for i in range(ind.size):
        print(csv_pos.loc[ind[i]])
        csv_pos.loc[ind[i]] = csv_pos2.loc[i]
        print(csv_pos.loc[ind[i]])
    csv_pos['userID'] = csv_pos['userID'].apply(literal_eval)
    csv_pos['userID'] = csv_pos['userID'].apply(lambda x: list(x.values())[0])
    csv_pos['restaurantID'] = csv_pos['restaurantID'].apply(literal_eval)
    csv_pos['restaurantID'] = csv_pos['restaurantID'].apply(lambda x: list(x.values())[0])
    csv_pos['rating'] = csv_pos['rating'].apply(literal_eval)
    csv_pos['rating'] = csv_pos['rating'].apply(lambda x: list(x.values())[0])

    csv_pos.to_csv("..\\tokenNeigh.csv", sep=';', header=False, index=False)
except Exception as e:
    print(e)