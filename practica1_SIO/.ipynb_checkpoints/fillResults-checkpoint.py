import pandas as pd
import numpy as np

try:
    my_csv = pd.read_csv('..\\predicted_neighbours.csv')
    template_csv = pd.read_csv('..\\PlantillaPrediccions.csv', delimiter=';', names=['userID', 'restaurantID', 'rating'])
    template_copy = template_csv.copy()
    template_copy.drop('rating', axis=1, inplace=True)
    
    for i in range(9):
        array = template_copy.iloc[i]
        print(array)
    filtered_df = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])
except Exception as e:
    print(e)

