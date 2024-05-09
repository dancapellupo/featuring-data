
from tqdm.auto import tqdm

import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler


def plot_ecdf(data_col, data_label='', xlabel='Data Values', filename='ecdf', overplot=False, outfile=True,
              plots_folder='./'):

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


def plot_hist(data_for_bins, label_bins='', data_for_line=None, label_line='', xlabel='Data Values', ylabel='Count',
              filename='hist', plots_folder='./'):

    sns.set_theme(style="ticks", font_scale=1.2)
    f, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(data=data_for_bins, bins=10, binrange=(0, 1), label=label_bins)
    if data_for_line is not None:
        sns.histplot(data=data_for_line, bins=10, binrange=(0, 1), element='step', fill=False, color='orange',
                     label=label_line)
    plt.xlim(0, 1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 1)
    plt.legend()

    plt.savefig('{}/{}.png'.format(plots_folder, filename), bbox_inches='tight')
    plt.close()


def plot_hist_target_col(target_col_vals, target_type='regression', plots_folder='./'):

    sns.set_theme(style="ticks", font_scale=1.2)
    f, ax = plt.subplots(figsize=(9, 6))

    if target_type == 'regression':
        sns.histplot(data=target_col_vals)
        plt.grid()
        plt.xlim(target_col_vals.min(), target_col_vals.max())
    else:
        sns.histplot(data=target_col_vals, discrete=True, shrink=0.6)
        ax.set_xticks(target_col_vals.unique())
        plt.grid(axis='y')

    plt.savefig('{}/target_data_distribution.png'.format(plots_folder), bbox_inches='tight')
    plt.close()


def plot_scatter_density(x, y, fig=None, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()

    try:
        bins = [bins, bins]
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                    method="splinef2d", bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

    except ValueError:
        # Calculate the point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    # z = MinMaxScaler(feature_range=(0, 1)).fit_transform(z.reshape(-1, 1))
    z /= 10**(math.floor(math.log10(abs(z.max()))))
    
    plt.scatter(x, y, c=z, **kwargs)
    plt.colorbar()

    # norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    # cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm), ax=ax, cmap='viridis')
    # cbar.ax.set_ylabel('Density')

    return ax


def plot_feature_values(data_df, columns_list, correlation_df, target_col, numeric=True, target_type='regression',
                        plot_style='scatterdense', plots_folder='./plots'):
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
        The options are:
        - 'scatterdense' for density scatterplots with the matplotlib viridis color palette
        - 'swarm' or 'strip' for default seaborn colors and style

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

    # Set box plot display parameters:
    if plot_style != 'scatterdense':
        box_params = {'whis': [0, 100], 'width': 0.6}
    else:
        box_params = {'whis': [0, 100], 'width': 0.6, 'fill': False, 'color': 'black'}

    set_ylim = False
    if target_type == 'regression':
        # Check for strong outliers in target column:
        med = data_df[target_col].median()
        std = data_df[target_col].std()
        xx = np.where(data_df[target_col].values > med + 10*std)[0]
        xx = np.append(xx, np.where(data_df[target_col].values < med - 10*std)[0])
        if xx.size > 0:
            print('Target outlier points:', data_df[target_col].values[xx])

            target_col_vals = data_df.reset_index().drop(xx)[target_col].values
            target_min, target_max = np.min(target_col_vals), np.max(target_col_vals)
            max_minus_min = target_max - target_min
            ymin = target_min - 0.025*max_minus_min
            ymax = target_max + 0.025*max_minus_min
            print('New target min/max values:', target_min, target_max)
            print('Set y-axis limits (for display only): {:.2f} {:.2f}.\n'.format(ymin, ymax))
            set_ylim = True

    sns.set_theme(style="ticks")

    print('Generating plots of {} features...'.format('numeric' if numeric else 'non-numeric/categorical'))
    for jj, column in enumerate(tqdm(columns_list)):

        f, ax = plt.subplots(figsize=(9, 6))

        data_df_col_notnull = data_df[[column, target_col]].dropna().reset_index()

        # TODO: User can define this value:
        num_uniq = correlation_df.loc[column, "Num Unique Values"]
        if (not numeric) or (num_uniq <= 10):

            if num_uniq > 20:
                orig_len = len(data_df_col_notnull)
                value_counts_index = data_df_col_notnull[column].value_counts().index[0:20]
                data_df_col_notnull = data_df_col_notnull.loc[data_df_col_notnull[column].isin(value_counts_index)]
                print("For '{}', more than 20 unique values: Only plotting top 20, which is {} out of {} total data"
                      "points.".format(column, len(data_df_col_notnull), orig_len))
                anc = AnchoredText('Plotting top 20 out of {} total uniq vals'.format(num_uniq), loc="upper left",
                                   frameon=False)
                ax.add_artist(anc)
            
            if target_type == 'regression':
                if not numeric:
                    # Standard Box Plot with X-axis ordered by median value in each category
                    xaxis_order = data_df_col_notnull.groupby(
                        by=[column]).median().sort_values(by=[target_col]).index.tolist()

                    sns.boxplot(data_df_col_notnull, x=column, y=target_col, order=xaxis_order, **box_params)

                else:
                    # Standard Box Plot
                    sns.boxplot(data_df_col_notnull, x=column, y=target_col, **box_params)  # hue="method", palette="vlag"

                # Add in points to show each observation
                if (plot_style != 'scatterdense') and (len(data_df_col_notnull) > 1000):
                    data_df_col_notnull = data_df_col_notnull.sample(n=1000, replace=False)

                if plot_style in ('swarm', 'seaborn'):
                    sns.swarmplot(data_df_col_notnull, x=column, y=target_col, size=2, color=".3", warn_thresh=0.4)

                elif plot_style == 'strip':
                    sns.stripplot(data_df_col_notnull, x=column, y=target_col, jitter=0.25, size=2, color=".3")

                elif plot_style == 'scatterdense':
                    x_all, y_all = np.array([]), np.array([])

                    for cat in ax.get_xticklabels():
                        # print(cat, cat.get_text(), cat.get_position(), cat.get_position()[0])

                        try:
                            data_df_cat = data_df_col_notnull.loc[
                                (data_df_col_notnull[column] == cat.get_text()) | (data_df_col_notnull[column] == float(cat.get_text()))]
                        except ValueError:
                            data_df_cat = data_df_col_notnull.loc[data_df_col_notnull[column] == cat.get_text()]
                        # print(len(data_df_cat))

                        x = (np.zeros(len(data_df_cat)) + cat.get_position()[0] +
                            np.random.normal(scale=0.06, size=len(data_df_cat)))  # 0.005
                        y = data_df_cat[target_col].values
                        x_all, y_all = np.append(x_all, x), np.append(y_all, y)

                    ax = plot_scatter_density(x_all, y_all, fig=f, ax=ax, bins=100, s=3, cmap='viridis')
                
            else:
                if plot_style == 'seaborn':
                    sns.histplot(data_df_col_notnull, x=column, hue=target_col, discrete=True, shrink=0.6,
                                 multiple="dodge")  # "stack"
                else:
                    with sns.color_palette('viridis'):
                        sns.histplot(data_df_col_notnull, x=column, hue=target_col, discrete=True, shrink=0.6,
                                     multiple="dodge")  # "stack"
                ax.set_xticks(data_df_col_notnull[column].unique())

            if (not numeric) and num_uniq >= 10:
                plt.xticks(rotation=45)
                plt.grid(axis='x')

            plt.grid(axis='y')

        else:

            med = data_df_col_notnull[column].median()
            std = data_df_col_notnull[column].std()
            xx = np.where(data_df_col_notnull[column].values > med + 10*std)[0]
            # print(xx)

            if xx.size > 0:
                data_df_col_notnull = data_df_col_notnull.drop(xx)

                anc = AnchoredText('Not Shown: {} Outliers'.format(xx.size), loc="upper left", frameon=False)
                ax.add_artist(anc)

            if target_type == 'regression':
                if plot_style == 'seaborn':
                    sns.scatterplot(data_df_col_notnull, x=column, y=target_col, size=2, legend=False)

                else:
                    ax = plot_scatter_density(data_df_col_notnull[column].values, data_df_col_notnull[target_col].values,
                                            fig=f, ax=ax, bins=100, s=3, cmap='viridis')

                    # plt.hist2d(data_df_col_notnull[column], data_df_col_notnull[target_col], bins=(100, 100),
                    #            cmap='viridis', cmin=1)  # BuPu
                    # plt.colorbar()

                    # ax.scatter(x, y, c=z, s=100, edgecolor='')
                    # ax.scatter(x, y, c=z, s=50)
            
            else:
                sns.boxplot(data_df_col_notnull, x=column, y=target_col, orient='y', **box_params)

                if (plot_style != 'scatterdense') and (len(data_df_col_notnull) > 1000):
                    data_df_col_notnull = data_df_col_notnull.sample(n=1000, replace=False)

                if plot_style in ('swarm', 'seaborn'):
                    sns.swarmplot(data_df_col_notnull, x=column, y=target_col, orient='y', size=2, color=".3", warn_thresh=0.4)

                elif plot_style == 'strip':
                    sns.stripplot(data_df_col_notnull, x=column, y=target_col, orient='y', jitter=0.25, size=2, color=".3")
                
                elif plot_style == 'scatterdense':
                    x_all, y_all = np.array([]), np.array([])

                    for cat in ax.get_yticklabels():
                        # print(cat, cat.get_text(), cat.get_position(), cat.get_position()[0])

                        try:
                            data_df_cat = data_df_col_notnull.loc[
                                (data_df_col_notnull[target_col] == cat.get_text()) | (
                                            data_df_col_notnull[target_col] == float(cat.get_text()))]
                        except ValueError:
                            data_df_cat = data_df_col_notnull.loc[data_df_col_notnull[target_col] == cat.get_text()]
                        # print(len(data_df_cat))

                        y = (np.zeros(len(data_df_cat)) + cat.get_position()[1] +
                            np.random.normal(scale=0.06, size=len(data_df_cat)))
                        x = data_df_cat[column].values
                        x_all, y_all = np.append(x_all, x), np.append(y_all, y)

                    ax = plot_scatter_density(x_all, y_all, fig=f, ax=ax, bins=100, s=3, cmap='viridis')

            plt.grid()

            plt.xlabel(column)
            plt.ylabel(target_col)

        if set_ylim:
            plt.ylim(ymin, ymax)

        if numeric:
            if target_type == 'regression':
                ax.set_title('{} vs {} : P={}, MI={}, RF={}'.format(
                    column, target_col, correlation_df.loc[column, "Pearson"],
                    correlation_df.loc[column, "Mutual Info"], correlation_df.loc[column, "Random Forest"]))
            else:
                ax.set_title('{} vs {} : MI={}, RF={}'.format(
                    column, target_col, correlation_df.loc[column, "Mutual Info"],
                    correlation_df.loc[column, "Random Forest"]))
        else:
            ax.set_title('{} vs {} : MI={}, RF={}, RF_norm={}'.format(
                column, target_col, correlation_df.loc[column, "Mutual Info"],
                correlation_df.loc[column, "Random Forest"], correlation_df.loc[column, "RF_norm"]))

        plt.savefig('{}/{}_vs_{}.png'.format(plots_folder, column, target_col), bbox_inches='tight')

        plt.close()

    # mpl.use(backend_)  # Reset backend
    # print('*** {} ***'.format(mpl.get_backend()))

