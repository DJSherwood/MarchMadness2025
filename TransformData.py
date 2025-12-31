# this is based on a solution by a bronze winner in 2025
# https://www.kaggle.com/code/modeh7/final-solution-ncaa-2025
# transforming pandas to polars
# transofrming xgboost to catboost
# possibly add conformal prediction

import numpy as np
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn
import warnings
warnings.filterwarnings('ignore')
data_dir = "./march-machine-learning-mania-2025/"


def _standardize(col, numOT):
    return (40 + 5 * numOT) / 40

class TransformData():
    def __init__(self):
        self.files_to_load = ['RegularSeasonDetailedResults', 'NCAATourneyDetailedResults', 'NCAATourneySeeds']
        self.df_list = []
        self.fl_data = []
        self.cols_to_stdize = ["LScore", "WScore","LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR",
                               "LAst", "LTO", "LStl", "LBlk", "LPF","WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
                               "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]

    def load_data(self):
        for f in self.files_to_load:
            male_temp = pl.read_csv(data_dir + 'M' + f + '.csv').with_columns(
                pl.lit(1).alias("men_women")
            )
            female_temp = pl.read_csv(data_dir + 'W' + f + '.csv').with_columns(
                pl.lit(0).alias("men_women")
            )
            self.df_list.append(pl.concat([male_temp, female_temp]))

    def filter_data(self, season=2024, teamid=1443):
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
        self.fl_data = pl.concat([temp_df_list[0], temp_df_list[1]])

    def prepare(self, df):
        df_prepped = df.select(
            ["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT", "LFGM",
             "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl",
             "LBlk", "LPF", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR",
             "WAst","WTO", "WStl", "WBlk", "WPF", "men_women"]
        ).with_columns(
            (
                    ( pl.col(self.cols_to_stdize) * 40 ) / ( 40 + 5 * pl.col("NumOT") )
            ).name.keep()
        )
        # rename columns and stack
        dfA = df_prepped.rename(lambda col_name: col_name.replace("W", "T1_").replace("L", "T2_"))
        dfB = df_prepped.rename(lambda col_name: col_name.replace("L", "T1_").replace("W", "T2_"))
        A_cols = dfA.columns
        dfB = dfB.select(A_cols)
        df_prepped = pl.concat([dfA, dfB])
        # create_features
        df_prepped = df_prepped.with_columns(
            (pl.col("T1_Score") - pl.col("T2_Score")).alias("PointDiff")
        ).with_columns(
            pl.when(pl.col("PointDiff") > 0).then(pl.col("PointDiff")).otherwise(0).alias("win")
        )
        return df_prepped

if __name__ == '__main__':
    td = TransformData()
    td.load_data()
    print(td.prepare(td.df_list[0]))
    # td.expl_data.glimpse()

