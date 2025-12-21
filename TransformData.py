# this is based on a solution by a bronze winner in 2025
# https://www.kaggle.com/code/modeh7/final-solution-ncaa-2025
# transforming pandas to polars
# transofrming xgboost to catboost
# possibly add conformal prediction

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn
import warnings
warnings.filterwarnings('ignore')
data_dir = "./march-machine-learning-mania-2025"

# static helper file
def _load_helper():
    list_to_load = []
    list_of_sexes = ['M', 'W']
    list_of_files = ['RegularSeasonDetailedResults', 'NCAATourneyDetailedResults', 'NCAATourneySeeds']
    for s in list_of_sexes:
        for f in list_of_files:
            list_to_load.append(data_dir + s + f + '.csv')

    return list_to_load

class TransformData():
    def __init__(self):
        self.files_to_load = _load_helper()
        self.df_list = []

    def load_data(self):
        for f in self.files_to_load:
            self.df_list.append(pl.read_csv(f))
