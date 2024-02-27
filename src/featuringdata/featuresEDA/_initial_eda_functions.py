
import numpy as np
import pandas as pd


def count_null_values(data_df):
    null_cols_df = pd.DataFrame(columns=["Feature", "Num of Nulls", "Frac Null"])

    null_cols = data_df.columns[data_df.isna().any()].tolist()

    for jj, col in enumerate(null_cols):
        num_nulls = data_df[col].isna().sum()
        null_cols_df.loc[jj] = col, num_nulls, round(num_nulls / len(data_df), 2)

    null_cols_df = null_cols_df.sort_values(by=["Num of Nulls"], ascending=False)

    return null_cols_df


def sort_numeric_nonnumeric_columns(data_df, target_col=None):

    numeric_cols = data_df.select_dtypes(include='number').columns.to_list()
    non_numeric_cols = data_df.select_dtypes(exclude='number').columns.to_list()

    if target_col is not None:
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        elif target_col in non_numeric_cols:
            non_numeric_cols.remove(target_col)

    print('There are {} numeric columns and {} non-numeric columns.'.format(len(numeric_cols),
                                                                            len(non_numeric_cols)))

    return numeric_cols, non_numeric_cols


def count_numeric_unique_values(data_df, numeric_cols, uniq_vals_thresh=10):

    numeric_uniq_vals_df = pd.DataFrame(columns=["Feature", "Num Unique Values"])

    jj = 0
    for col in numeric_cols:
        num_uniq = np.unique(data_df[col]).size

        if num_uniq <= uniq_vals_thresh:
            numeric_uniq_vals_df.loc[jj] = col, num_uniq
            jj += 1

    numeric_uniq_vals_df = numeric_uniq_vals_df.sort_values(by=["Num Unique Values"])

    return numeric_uniq_vals_df


def count_nonnumeric_unique_values(data_df, non_numeric_cols, uniq_vals_thresh=5):

    non_numeric_uniq_vals_df = pd.DataFrame(columns=["Feature", "Num Unique Values"])

    jj = 0
    for col in non_numeric_cols:
        num_uniq = data_df[col].nunique()

        # if num_uniq > uniq_vals_thresh:
        non_numeric_uniq_vals_df.loc[jj] = col, num_uniq
        jj += 1

    non_numeric_uniq_vals_df = non_numeric_uniq_vals_df.sort_values(by=["Num Unique Values"], ascending=False)

    return non_numeric_uniq_vals_df


