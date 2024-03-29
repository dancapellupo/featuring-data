
from tqdm.auto import tqdm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns


def plot_ecdf(data_col, data_label='', xlabel='Data Values', filename='ecdf', overplot=False, outfile=True, plots_folder='./'):

    if not overplot:
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots(figsize=(8, 5))

    sns.ecdfplot(data=data_col, complementary=True, label=data_label)

    if outfile:
        plt.xlabel(xlabel)
        plt.xlim(0, 1)
        plt.legend()

        plt.savefig('{}/{}.png'.format(plots_folder, filename), bbox_inches='tight')
        plt.close()


def plot_feature_values(data_df, columns_list, correlation_df, target_col, numeric=True, catplot_style='swarm',
                        plots_folder='./plots'):
    """
    Generate EDA plots that show each feature versus the target variable.

    The code automatically adjusts based on certain properties of the feature:
    - For categorical features, as well as numeric features with up to 10
      unique values, a box plot with a swarm plot is generated. If there are
      more than 1,000 data points, then only a random selection of 1,000
      points are plotted on the swarm plot (but the box plot is calculated
      based on all points).
    - For typical numeric features, a standard scatter plot is generated. Any
      large outliers, located more than 10 standard deviations from the
      median, are not shown.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input dataframe.

    columns_list : list
        A list of column names to plot.

    correlation_df : pd.DataFrame
        A dataframe with measures of the correlation of each feature with the
        target variable. The dataframe is the output from either
        '_correlation.calc_numeric_features_target_corr' or
        '_correlation.calc_nonnumeric_features_target_corr'.

    target_col : str

    numeric : bool

    catplot_style : str

    plots_folder : str

    Returns
    -------
    r2 : float
        The theoretical maximum R^2 for the given number of unique values.
    """

    # backend_ = mpl.get_backend()
    # print('*** {} ***'.format(backend_))
    # mpl.use("Agg")
    # print('*** {} ***'.format(mpl.get_backend()))

    if len(data_df) > 1000:
        data_df_sample = data_df.sample(n=1000, replace=False)

    sns.set_theme(style="ticks")

    print('Generating plots of {} features...'.format('numeric' if numeric else 'non-numeric/categorical'))
    for jj, column in enumerate(tqdm(columns_list)):

        f, ax = plt.subplots(figsize=(9, 6))

        # TODO: Use already calculated DF of unique values:
        # TODO: User can define this value:
        if (not numeric) or (np.unique(data_df[column]).size <= 10):

            if not numeric:
                # Standard Box Plot with X-axis ordered by median value in each category
                data_df_col_notnull = data_df[[column, target_col]].dropna()
                xaxis_order = data_df_col_notnull.groupby(
                    by=[column]).median().sort_values(by=[target_col]).index.tolist()

                sns.boxplot(data_df_col_notnull, x=column, y=target_col, order=xaxis_order, whis=[0, 100], width=0.6)

            else:
                # Standard Box Plot
                sns.boxplot(data_df, x=column, y=target_col, whis=[0, 100], width=0.6,)  # hue="method", palette="vlag"

            # Add in points to show each observation
            if catplot_style == 'swarm':
                if 'data_df_sample' in locals():
                    sns.swarmplot(data_df_sample, x=column, y=target_col, size=2, color=".3", warn_thresh=0.4)
                else:
                    sns.swarmplot(data_df, x=column, y=target_col, size=2, color=".3", warn_thresh=0.4)
            else:
                sns.stripplot(data_df, x=column, y=target_col, jitter=0.25, size=2, color=".3")

            if (not numeric) and data_df[column].nunique() >= 10:
                plt.xticks(rotation=45)

        else:
            med = data_df[column].median()
            std = data_df[column].std()
            xx = np.where(data_df[column].values > med + 10*std)[0]
            # print(xx)

            if xx.size == 0:
                # sns.scatterplot(train_data_mod, x=column, y=target_col, hue="OverallQual")
                sns.scatterplot(data_df, x=column, y=target_col, size=2, legend=False)
            else:
                # TODO Need to check index for using xx here
                sns.scatterplot(data_df.drop(xx), x=column, y=target_col, size=2, legend=False)
                anc = AnchoredText('Not Shown: {} Outliers'.format(xx.size), loc="upper left", frameon=False)
                ax.add_artist(anc)

        if numeric:
            ax.set_title('{} vs {} : P={}, MI={}, RF={}'.format(
                target_col, column, correlation_df.loc[column, "Pearson"],
                correlation_df.loc[column, "Mutual Info"], correlation_df.loc[column, "Random Forest"]))
        else:
            ax.set_title('{} vs {} : RF={}, RF_norm={}'.format(
                target_col, column, correlation_df.loc[column, "Random Forest"],
                correlation_df.loc[column, "RF_norm"]))

        plt.savefig('{}/{}_vs_{}.png'.format(plots_folder, column, target_col), bbox_inches='tight')

        plt.close()

    # mpl.use(backend_)  # Reset backend
    # print('*** {} ***'.format(mpl.get_backend()))

