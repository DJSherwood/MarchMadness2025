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
data_dir = "./march-machine-learning-mania-2025/"

class TransformData():
    def __init__(self):
        self.files_to_load = ['RegularSeasonDetailedResults', 'NCAATourneyDetailedResults', 'NCAATourneySeeds']
        self.df_list = []
        self.expl_data = []

    def load_data(self):
        for f in self.files_to_load:
            male_temp = pl.read_csv(data_dir + 'M' + f + '.csv')
            female_temp = pl.read_csv(data_dir + 'W' + f + '.csv')
            self.df_list.append(pl.concat([male_temp, female_temp]))

    def transform_data(self, season=2024, teamid=1443):
        # filter regular results for exploration
        temp_df_list = []
        for i in range(2):
            temp = self.df_list[i].filter(
                pl.col("Season") == season,
                ( pl.col("WTeamID") == teamid ) | ( pl.col("LTeamID") == teamid )
            )
            temp = temp.with_columns(
                pl.when( pl.col("WTeamID") == teamid).then(1).otherwise(0).alias("win")
            )
            if i == 0:
                temp = temp.with_columns(pl.lit("r").alias("type"))
            else:
                temp = temp.with_columns(pl.lit("t").alias("type"))
            temp_df_list.append(temp)
        self.expl_data = pl.concat([temp_df_list[0], temp_df_list[1]])


if __name__ == '__main__':
    td = TransformData()
    td.load_data()
    # td.transform_data()
    # td.expl_data.glimpse()

