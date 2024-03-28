
import numpy as np
import pandas as pd


def count_null_values(data_df):
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

    # Sort the dataframe by number of NULL values per feature, in descending order:
    null_cols_df = null_cols_df.sort_values(by=["Num of Nulls"], ascending=False)

    # Count the number of NULL values in each row:
    null_count_by_row_series = data_df.isna().sum(axis=1)

    return null_cols_df, null_count_by_row_series


def sort_numeric_nonnumeric_columns(data_df, target_col=None):
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


