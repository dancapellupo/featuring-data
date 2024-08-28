
import numpy as np
import pandas as pd

from ..utils import round_to_n_sigfig


def count_null_values(data_df, master_columns_df):
    """
    Counts the null values for every column in the input dataframe.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input dataframe.

    Returns
    -------
    null_cols_df : pd.DataFrame
        A dataframe with all the features that have at least one NULL value.
        The dataframe has the following columns:
        - "Feature": Name of the data column.
        - "Num of Nulls": Total number of null values in the column.
        - "Frac Null": The fraction of all values in that column that are null.
    """
    null_cols_df = pd.DataFrame(columns=["Feature", "Num of Nulls", "Frac Null"])

    null_cols = data_df.columns[data_df.isna().any()].tolist()

    for jj, col in enumerate(null_cols):
        num_nulls = data_df[col].isna().sum()
        null_cols_df.loc[jj] = col, num_nulls, round(num_nulls / len(data_df), 2)

    master_columns_df["Num of Nulls"] = data_df.isna().sum()
    master_columns_df["Frac Null"] = (master_columns_df["Num of Nulls"] / len(data_df)).round(2)

    # Sort the dataframe by number of NULL values per feature, in descending order:
    null_cols_df = null_cols_df.sort_values(by=["Num of Nulls"], ascending=False)

    # Count the number of NULL values in each row:
    null_count_by_row_series = data_df.isna().sum(axis=1)

    return master_columns_df, null_count_by_row_series


def sort_numeric_nonnumeric_columns(data_df, master_columns_df, target_col=None):
    """
    Sorts the names of the numeric and nun-numeric/categorical columns into
    two separate lists.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input dataframe.

    target_col : str, default=None
        The name of the dataframe column containing the target variable.

    Returns
    -------
    numeric_cols : List
        A list of names of columns with numeric values.

    non_numeric_cols : List
        A list of names of columns with non-numeric / categorical values.
    """

    master_columns_df["dtype"] = data_df.dtypes
    master_columns_df["Column Type (orig)"] = master_columns_df["dtype"].apply(
        lambda x: 'non-numeric' if pd.api.types.is_string_dtype(x) else 'numeric')
    # master_columns_df["Column Type (orig)"] = [
    #     'non-numeric' if pd.api.types.is_string_dtype(data_df[col]) else 'numeric' for col in master_columns_df.index]

    numeric_cols = data_df.select_dtypes(include='number').columns.to_list()
    non_numeric_cols = data_df.select_dtypes(exclude='number').columns.to_list()

    if target_col is not None:
        master_columns_df.loc[target_col, "Column Type (orig)"] = 'target'
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        elif target_col in non_numeric_cols:
            non_numeric_cols.remove(target_col)

    print('There are {} numeric columns and {} non-numeric columns.'.format(
        len(numeric_cols), len(non_numeric_cols)))

    master_columns_df["Num Unique Values"] = data_df.nunique()

    def unique_values_issues(col_type_orig, num_uniq):
        if num_uniq == 1:
            return 'remove'
        elif (col_type_orig == 'numeric') and (num_uniq == 2):
            return 'switch to non-numeric'
        elif (col_type_orig == 'non-numeric') and (num_uniq > 0.1*len(data_df)):
            return 'remove'

    master_columns_df["Column Note"] = master_columns_df.apply(lambda x: unique_values_issues(x["Column Type (orig)"], x["Num Unique Values"]), axis=1)

    def update_col_type(col_type_orig, col_note):
        if col_note == 'switch to non-numeric':
            return 'non-numeric'
        elif col_note != 'remove':
            return col_type_orig
    
    master_columns_df["Column Type"] = master_columns_df.apply(lambda x: update_col_type(x["Column Type (orig)"], x["Column Note"]), axis=1)

    return master_columns_df


def count_numeric_unique_values(data_df, numeric_cols, uniq_vals_thresh=10):
    """
    Counts the number of unique values in every numeric column.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input dataframe.

    numeric_cols : List
        A list of names of columns with numeric values (from the function
        'sort_numeric_nonnumeric_columns').

    uniq_vals_thresh : int, default=10
        Any feature with fewer than this number of unique values will be saved
        to the output dataframe.

    Returns
    -------
    numeric_uniq_vals_df : pd.DataFrame
        A dataframe listing any numeric columns that have no more than
        "numeric_uniq_vals_thresh" unique values. The dataframe has the
        following columns:
        - "Feature": Name of the numeric data column.
        - "Num Unique Values": The number of unique values in that data
          column.
    """
    numeric_uniq_vals_df = pd.DataFrame(columns=["Feature", "Num Unique Values"])

    jj = 0
    for col in numeric_cols:
        num_uniq = np.unique(data_df[col]).size

        if num_uniq <= uniq_vals_thresh:
            numeric_uniq_vals_df.loc[jj] = col, num_uniq
            jj += 1

    # Sort the dataframe by number of unique values per feature, in ascending order:
    numeric_uniq_vals_df = numeric_uniq_vals_df.sort_values(by=["Num Unique Values"])

    return numeric_uniq_vals_df


def count_nonnumeric_unique_values(data_df, non_numeric_cols, uniq_vals_thresh=5):
    """
    Counts the number of unique values in every non-numeric column.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input dataframe.

    non_numeric_cols : List
        A list of names of columns with non-numeric values (from the function
        'sort_numeric_nonnumeric_columns').

    uniq_vals_thresh : int, default=10
        [Currently unused]
        Any non-numeric feature with greater than this number of unique values
        will be reported.

    Returns
    -------
    numeric_uniq_vals_df : pd.DataFrame
        A dataframe listing the number of unique values in every non-numeric
        column. The dataframe has the following columns:
        - "Feature": Name of the non-numeric data column.
        - "Num Unique Values": The number of unique values in that data
          column.
    """
    non_numeric_uniq_vals_df = pd.DataFrame(columns=["Feature", "Num Unique Values"])

    jj = 0
    for col in non_numeric_cols:
        num_uniq = data_df[col].nunique()

        # if num_uniq > uniq_vals_thresh:
        non_numeric_uniq_vals_df.loc[jj] = col, num_uniq
        jj += 1

    # Sort the dataframe by number of unique values per feature, in descending order:
    non_numeric_uniq_vals_df = non_numeric_uniq_vals_df.sort_values(by=["Num Unique Values"], ascending=False)

    return non_numeric_uniq_vals_df


def calc_column_summary_stats(data_df, master_columns_df):

    for jj in range(len(master_columns_df)):
        column = master_columns_df.index[jj]

        if master_columns_df["Column Type"].iloc[jj] == 'numeric':
            master_columns_df.loc[column, "Mean"] = round_to_n_sigfig(data_df[column].mean(), 4)
            master_columns_df.loc[column, "STD"] = round_to_n_sigfig(data_df[column].std(), 4)

            col_perc_tup = np.percentile(data_df[column].values, [0, 25, 50, 75, 100])
            master_columns_df.loc[column, "Min"] = col_perc_tup[0]
            master_columns_df.loc[column, "25th Perc"] = col_perc_tup[1]
            master_columns_df.loc[column, "50th Perc"] = col_perc_tup[2]
            master_columns_df.loc[column, "75th Perc"] = col_perc_tup[3]
            master_columns_df.loc[column, "Max"] = col_perc_tup[4]

        elif master_columns_df["Column Type"].iloc[jj] == 'non-numeric':
            column_describe = data_df[column].describe()
            master_columns_df.loc[column, "Most Frequent"] = column_describe.loc["top"]
            master_columns_df.loc[column, "Most Frequent Count"] = column_describe.loc["freq"]

    return master_columns_df

