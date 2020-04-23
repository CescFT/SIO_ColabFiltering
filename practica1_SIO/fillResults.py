import pandas as pd
import numpy as np

try:
    my_csv = pd.read_csv('..\\predicted_neighbours.csv')
    template_csv = pd.read_csv('..\\PlantillaPrediccions.csv', delimiter=';',
                               names=['userID', 'restaurantID', 'rating'])
    template_copy = template_csv.copy()
    template_copy.drop('rating', axis=1, inplace=True)

    final_df = pd.DataFrame(columns=['userID', 'restaurantID', 'rating'])
    for i in range(template_copy['userID'].size):
        print(str(i)+':'+str(template_copy['userID'].size))
        array = np.asarray(template_copy.iloc[i])
        row_to_fill = my_csv.loc[(my_csv['userID'] == array[0]) & (my_csv['restaurantID'] == array[1])]
        final_df = pd.concat([final_df, row_to_fill])
    final_df.to_csv('..\\tokenName.csv', index=False, header=False)

except Exception as e:
    print(e)
