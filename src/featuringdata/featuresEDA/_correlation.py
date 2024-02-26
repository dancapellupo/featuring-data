
import itertools as it

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from sklearn.ensemble import RandomForestRegressor


def calc_numeric_features_target_corr(data_df, numeric_cols, target_col, rf_n_estimators=10):

    numeric_df = pd.DataFrame(columns=["Count not-Null", "Pearson", "Mutual Info", "Random Forest"])

    for col in numeric_cols:
        data_df_col_notnull = data_df[[col, target_col]].dropna()

        pcorr = pearsonr(data_df_col_notnull[col].values, data_df_col_notnull[target_col].values)[0]
        minfo = mutual_info_regression(data_df_col_notnull[col].values.reshape(-1, 1), data_df_col_notnull[target_col].values)[0]

        rf_reg = RandomForestRegressor(n_estimators=10)
        rf_reg.fit(data_df_col_notnull[col].values.reshape(-1, 1), data_df_col_notnull["SalePrice"].values)
        rfscore = rf_reg.score(data_df_col_notnull[col].values.reshape(-1, 1), data_df_col_notnull["SalePrice"])

        numeric_df.loc[col] = len(data_df_col_notnull), round(pcorr, 2), round(minfo, 2), round(rfscore, 2)

    numeric_df["Count not-Null"] = numeric_df["Count not-Null"].astype(int)

    numeric_df = numeric_df.sort_values(by=["Random Forest"], ascending=False)

    return numeric_df


def calc_corr_numeric_features(data_df, numeric_cols):

    numeric_collinear_df = pd.DataFrame(columns=["Feature1", "Feature2", "Count not-Null", "Pearson", "Random Forest"])

    # jj = 0
    # for pair in it.combinations(numeric_cols, 2):
    #     col1, col2 = pair[0], pair[1]
    #
    #     data_df_cols_notnull = data_df[[col1, col2]].dropna()
    #
    #     pcorr = pearsonr(data_df_cols_notnull[col1].values, data_df_cols_notnull[col2].values)[0]
    #
    #     rf_reg = RandomForestRegressor(n_estimators=10)
    #     rf_reg.fit(data_df_cols_notnull[col1].values.reshape(-1, 1), data_df_cols_notnull[col2].values)
    #     rfscore1 = rf_reg.score(data_df_cols_notnull[col1].values.reshape(-1, 1), data_df_cols_notnull[col2])
    #
    #     rf_reg = RandomForestRegressor(n_estimators=10)
    #     rf_reg.fit(data_df_cols_notnull[col2].values.reshape(-1, 1), data_df_cols_notnull[col1].values)
    #     rfscore2 = rf_reg.score(data_df_cols_notnull[col2].values.reshape(-1, 1), data_df_cols_notnull[col1])
    #
    #     numeric_collinear_df.loc[jj] = col1, col2, len(data_df_cols_notnull), round(pcorr, 2), round((rfscore1+rfscore2)/2, 2)
    #
    #     jj += 1
    #
    # numeric_collinear_df = numeric_collinear_df.sort_values(by=["Random Forest"], ascending=False)

    for jj, pair in enumerate(it.permutations(numeric_cols, 2)):
        col1, col2 = pair[0], pair[1]

        data_df_cols_notnull = data_df[[col1, col2]].dropna()

        pcorr = pearsonr(data_df_cols_notnull[col1].values, data_df_cols_notnull[col2].values)[0]

        rf_reg = RandomForestRegressor(n_estimators=10)
        rf_reg.fit(data_df_cols_notnull[col1].values.reshape(-1, 1), data_df_cols_notnull[col2].values)
        rfscore = rf_reg.score(data_df_cols_notnull[col1].values.reshape(-1, 1), data_df_cols_notnull[col2])

        numeric_collinear_df.loc[jj] = col1, col2, len(data_df_cols_notnull), round(pcorr, 2), round(rfscore, 2)

    numeric_collinear_summary_df = pd.DataFrame(
        columns=["Avg Pearson Corr", "Avg RF Corr", "Max Corr Feature", "Max RF Corr"])

    for col in numeric_cols:
        numeric_collinear_df_col = numeric_collinear_df.loc[numeric_collinear_df["Feature1"] == col]

        rf_xx = np.argmax(numeric_collinear_df_col["Random Forest"].values)

        numeric_collinear_summary_df.loc[col] = (
            numeric_collinear_df_col["Pearson"].mean(), numeric_collinear_df_col["Random Forest"].mean(),
            numeric_collinear_df_col["Feature2"].iloc[rf_xx], numeric_collinear_df_col["Random Forest"].iloc[rf_xx])

    return numeric_collinear_df, numeric_collinear_summary_df


def calc_max_rfscore(num=2):
    y = np.arange(0, 1.001, 0.001)
    y_pred = np.zeros(1001)

    split = 1 / float(num)
    split_mid = split / 2

    for jj in range(num):
        if jj < num - 1:
            y_pred[int(jj*split*1000):int((jj+1)*split*1000)] = split_mid + jj*split
        else:
            y_pred[int(jj*split*1000):] = split_mid + jj*split

    r2 = (1 - ((y - y_pred)**2).sum() / ((y - 0.5)**2).sum())

    return r2


def calc_nonnumeric_features_target_corr(data_df, non_numeric_cols):

    non_numeric_df = pd.DataFrame(columns=["Count not-Null", "Num Unique", "Random Forest", "RF_norm"])

    for col in non_numeric_cols:
        train_col_notnull = data_df[[col, "SalePrice"]].dropna()

        X_col = pd.get_dummies(train_col_notnull[col], dtype=int)

        rf_reg = RandomForestRegressor(n_estimators=10)
        rf_reg.fit(X_col, train_col_notnull["SalePrice"].values)
        rfscore = rf_reg.score(X_col, train_col_notnull["SalePrice"])

        num_uniq = train_col_notnull[col].nunique()
        rfscore_norm = rfscore * (1 / calc_max_rfscore(num_uniq))

        non_numeric_df.loc[col] = len(train_col_notnull), num_uniq, round(rfscore, 2), round(rfscore_norm, 2)

    non_numeric_df["Count not-Null"] = non_numeric_df["Count not-Null"].astype(int)
    non_numeric_df["Num Unique"] = non_numeric_df["Num Unique"].astype(int)

    non_numeric_df = non_numeric_df.sort_values(by=["RF_norm"], ascending=False)

    return non_numeric_df

