# this is based on a solution by a bronze winner in 2025
# https://www.kaggle.com/code/modeh7/final-solution-ncaa-2025
# transforming pandas to polars
# transofrming xgboost to catboost
# possibly add conformal prediction

import numpy as np
import polars as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tqdm
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
        self.tourney_data = []
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

    def _prepare(self, flag=1):
        df_prepped = self.df_list[flag].select(
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

    def _prepare_seeds(self):
        dfSeeds = self.df_list[2].with_columns(
            pl.col("Seed").str.slice(1).cast(pl.Int32).alias("seed")
        )
        return dfSeeds

    def transform_tourney(self):
        # seed data
        seeds_T1 = self._prepare_seeds().select(["Season", "TeamID", "seed"]).rename(["Season", "T1_TeamID", "T1_seed"])
        seeds_T2 = self._prepare_seeds().select(["Season", "TeamID", "seed"]).rename(["Season", "T2_TeamID", "T2_seed"])
        # tourney data - why doesn't the flag argument work?
        tourney_data = self._prepare(self, flag=1).select(["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"])
        tourney_data = tourney_data.join(seeds_T1, on=["Season", "T1_TeamID"], how="left")
        tourney_data = tourney_data.join(seeds_T2, on=["Season", "T2_TeamID"], how="left")
        self.tourney_data = tourney_data.with_columns(
            (pl.col("T2_seed") - pl.col("T1_seed")).alias("Seed_Diff")
        )
        return self

    def merge_season_averages(self):
        """
        Calculate regular season box score averages per team and merge onto tourney data.
        T1_ columns become T1_avg_* and T2_ (opponent) columns become T1_avg_opponent_*.
        """
        boxcols = [
            "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
            "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
            "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
            "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
            "PointDiff",
        ]

        regular_data = self._prepare(flag=0)

        # Season averages grouped by Season + T1_TeamID
        ss = (
            regular_data
            .group_by(["Season", "T1_TeamID"])
            .agg([pl.col(c).mean() for c in boxcols])
        )

        def rename_cols(col: str, prefix: str, team_id_alias: str) -> str:
            if col == "Season":
                return "Season"
            if col == "T1_TeamID":
                return team_id_alias
            col = col.replace("T1_", "").replace("T2_", "opponent_")
            return prefix + col

        # Build ss_T1: averages from T1's perspective
        ss_T1 = ss.rename({
            col: rename_cols(col, prefix="T1_avg_", team_id_alias="T1_TeamID")
            for col in ss.columns
        })

        # Build ss_T2: same averages, re-labeled from T2's perspective
        ss_T2 = ss.rename({
            col: rename_cols(col, prefix="T2_avg_", team_id_alias="T2_TeamID")
            for col in ss.columns
        })

        # Merge onto tourney data
        tourney_data = (
            self.tourney_data
            .join(ss_T1, on=["Season", "T1_TeamID"], how="left")
            .join(ss_T2, on=["Season", "T2_TeamID"], how="left")
        )

        self.tourney_data = tourney_data
        return self

    @staticmethod
    def _expected_result(elo_a: float, elo_b: float, elo_width: int = 400) -> float:
        return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))

    @staticmethod
    def _update_elo(winner_elo: float, loser_elo: float, k_factor: int = 100, elo_width: int = 400):
        expected_win = TransformData._expected_result(winner_elo, loser_elo, elo_width)
        change_in_elo = k_factor * (1 - expected_win)
        return winner_elo + change_in_elo, loser_elo - change_in_elo

    def compute_elo(self, base_elo: int = 1000, elo_width: int = 400, k_factor: int = 100):
        """
        Compute end-of-season Elo ratings from regular season results,
        then merge T1 and T2 Elos onto tourney_data.
        """
        regular_data = self._prepare(flag=0)

        # Only keep wins (avoids double-counting — each game appears twice in _prepare output)
        wins_only = regular_data.filter(pl.col("win") > 0)

        seasons = (
            self._prepare_seeds()
            .select("Season")
            .unique()
            .sort("Season")
            .to_series()
            .to_list()
        )

        elos_list = []

        for season in seasons:
            ss = (
                wins_only
                .filter(pl.col("Season") == season)
                .select(["T1_TeamID", "T2_TeamID"])
                .to_numpy()  # convert once for fast row iteration
            )

            # Initialise every team in this season to base_elo
            teams = set(ss[:, 0]) | set(ss[:, 1])
            elo = dict.fromkeys(teams, float(base_elo))

            for w_team, l_team in ss:
                w_elo_new, l_elo_new = TransformData._update_elo(
                    elo[w_team], elo[l_team], k_factor, elo_width
                )
                elo[w_team] = w_elo_new
                elo[l_team] = l_elo_new

            elos_list.append(
                pl.DataFrame({
                    "TeamID": list(elo.keys()),
                    "elo": list(elo.values()),
                    "Season": season,
                })
            )

        elos = pl.concat(elos_list)

        # Merge onto tourney_data for both teams, then compute diff
        self.tourney_data = (
            self.tourney_data
            .join(
                elos.rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}),
                on=["Season", "T1_TeamID"],
                how="left"
            )
            .join(
                elos.rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}),
                on=["Season", "T2_TeamID"],
                how="left"
            )
            .with_columns(
                (pl.col("T1_elo") - pl.col("T2_elo")).alias("elo_diff")
            )
        )

        return self

    def compute_glm_quality(self, min_season_men: int = 2003, min_season_women: int = 2010):
        """
        Fit a Gaussian GLM per season/gender to derive a team quality rating
        from regular season PointDiff. Merges T1_quality and T2_quality onto tourney_data.
        """


        regular_data = self._prepare(flag=0)

        # --- Build Season/TeamID composite keys ---
        regular_data = regular_data.with_columns([
            (pl.col("Season").cast(pl.Utf8) + "/" + pl.col("T1_TeamID").cast(pl.Utf8)).alias("ST1"),
            (pl.col("Season").cast(pl.Utf8) + "/" + pl.col("T2_TeamID").cast(pl.Utf8)).alias("ST2"),
        ])

        seeds = self._prepare_seeds()
        seeds_T1 = seeds.rename({"TeamID": "T1_TeamID", "seed": "T1_seed"}).with_columns(
            (pl.col("Season").cast(pl.Utf8) + "/" + pl.col("T1_TeamID").cast(pl.Utf8)).alias("ST1")
        )
        seeds_T2 = seeds.rename({"TeamID": "T2_TeamID", "seed": "T2_seed"}).with_columns(
            (pl.col("Season").cast(pl.Utf8) + "/" + pl.col("T2_TeamID").cast(pl.Utf8)).alias("ST2")
        )

        # --- Collect tourney teams + non-tourney teams that beat a tourney team ---
        st_T1 = set(seeds_T1.get_column("ST1").to_list())
        st_T2 = set(seeds_T2.get_column("ST2").to_list())
        st = st_T1 | st_T2

        # Non-tourney teams that beat a tourney team at least once
        upset_qualifiers = (
            regular_data
            .filter(
                (pl.col("T1_Score") > pl.col("T2_Score")) &
                pl.col("ST2").is_in(st)
            )
            .get_column("ST1")
            .to_list()
        )
        st = st | set(upset_qualifiers)

        # --- Filter regular data to relevant teams; mask non-st teams as "0000" ---
        dt = (
            regular_data
            .filter(pl.col("ST1").is_in(st) | pl.col("ST2").is_in(st))
            .with_columns([
                pl.when(pl.col("ST1").is_in(st))
                .then(pl.col("T1_TeamID").cast(pl.Utf8))
                .otherwise(pl.lit("0000"))
                .alias("T1_TeamID"),
                pl.when(pl.col("ST2").is_in(st))
                .then(pl.col("T2_TeamID").cast(pl.Utf8))
                .otherwise(pl.lit("0000"))
                .alias("T2_TeamID"),
            ])
        )

        # Convert to pandas once — statsmodels requires it
        dt_pd = dt.to_pandas()

        # --- GLM helper (unchanged logic, operates on pandas slice) ---
        def team_quality(season: int, men_women: int):
            formula = "PointDiff ~ -1 + T1_TeamID + T2_TeamID"
            subset = dt_pd[(dt_pd["Season"] == season) & (dt_pd["men_women"] == men_women)]
            glm = sm.GLM.from_formula(
                formula=formula,
                data=subset,
                family=sm.families.Gaussian(),
            ).fit()

            quality = glm.params.reset_index()
            quality.columns = ["TeamID", "quality"]
            quality["Season"] = season
            quality = quality[quality["TeamID"].str.contains("T1_")].reset_index(drop=True)
            quality["TeamID"] = quality["TeamID"].str[10:14].astype(int)
            return quality

        # --- Loop over seasons ---
        seasons = sorted(seeds.get_column("Season").unique().to_list())
        glm_quality = []

        for s in tqdm(seasons, unit="season"):
            if s >= min_season_women:
                glm_quality.append(team_quality(s, men_women=0))
            if s >= min_season_men:
                glm_quality.append(team_quality(s, men_women=1))

        # --- Convert results back to Polars and merge ---
        import pandas as pd
        glm_quality_pd = pd.concat(glm_quality).reset_index(drop=True)
        glm_quality_pl = pl.from_pandas(glm_quality_pd)

        glm_T1 = glm_quality_pl.rename({"TeamID": "T1_TeamID", "quality": "T1_quality"})
        glm_T2 = glm_quality_pl.rename({"TeamID": "T2_TeamID", "quality": "T2_quality"})

        self.tourney_data = (
            self.tourney_data
            .join(glm_T1, on=["Season", "T1_TeamID"], how="left")
            .join(glm_T2, on=["Season", "T2_TeamID"], how="left")
            .with_columns(
                (pl.col("T1_quality") - pl.col("T2_quality")).alias("diff_quality")
            )
        )

        return self

if __name__ == '__main__':
    td = TransformData()
    td.load_data()
    print(td.prepare(td.df_list[0]))
    # td.expl_data.glimpse()

