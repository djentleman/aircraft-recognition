import pandas as pd
import os
from model import (
    preprocess_image
)

dataset = []
for filename in os.listdir('data'): # loop through all the files and folders
    print(filename)
    if os.path.isdir(os.path.join(os.path.abspath("data"), filename)): # check whether the current object is a folder or not
        path = 'data/' + filename
        for img in os.listdir(path):
            feature_vector = preprocess_image(path + '/' + img)
            dataset.append(list(feature_vector) + [filename])

dataset_df = pd.DataFrame(dataset)
dataset_df.to_csv('data/training_data.csv')




