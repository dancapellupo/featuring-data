
from tqdm.auto import tqdm

import math
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
try:
    from scipy.interpolate import interpn
    from scipy.stats import gaussian_kde
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    pass


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


def plot_hist_target_col(target_col_vals, target_type='regression', inline=False, inweb=False, set_plot_order=None, plots_folder='./'):

    if inline:
        sns.set_theme(style="ticks", font_scale=0.8)
        f, ax = plt.subplots(figsize=(3, 2))
        plt.title('Target Column Distribution')
    else:
        if inweb:
            sns.set_theme(style="ticks", font_scale=1.6)
        else:
            sns.set_theme(style="ticks", font_scale=1.2)
        f, ax = plt.subplots(figsize=(9, 6))

    if target_type == 'regression':
        sns.histplot(data=target_col_vals)
        plt.grid()
        plt.xlim(target_col_vals.min(), target_col_vals.max())
    else:
        if set_plot_order is not None:
            target_col_vals_cat = pd.Categorical(target_col_vals, set_plot_order)
            sns.histplot(data=target_col_vals_cat, discrete=True, shrink=0.6)
        else:
            sns.histplot(data=target_col_vals, discrete=True, shrink=0.6)
            # ax.set_xticks(target_col_vals.unique())
        plt.grid(axis='y')
        
        if target_col_vals.nunique() > 5:
            plt.xticks(rotation=45)

    if inline:
        plt.show()
    elif not inweb:
        plt.savefig('{}/target_data_distribution.png'.format(plots_folder), bbox_inches='tight')
        plt.close()


def plot_scatter_density(x, y, fig=None, ax=None, sort=True, bins=120, x_scale=120, y_scale=120, r2=0.8, inline=None, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()

    if x.shape[0] <= 10000:
        x_scaled = (x - x.min()) / (x.max() - x.min()) * x_scale
        y_scaled = (y - y.min()) / (y.max() - y.min()) * y_scale

        z = np.ones(x.size)
        for ii in range(x.size):
            z[ii] = np.where((x_scaled[ii] - x_scaled) ** 2 + (y_scaled[ii] - y_scaled) ** 2 <= r2)[0].size

    else:
        # try:
        #     bins = [bins, bins]
        #     data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        #     z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
        #                 method="splinef2d", bounds_error=False)
        #
        #     # To be sure to plot all data
        #     z[np.where(np.isnan(z))] = 0.0
        #
        # except ValueError:
        #     # Calculate the point density
        #     xy = np.vstack([x, y])
        #     z = gaussian_kde(xy)(xy)
        #
        # # z = MinMaxScaler(feature_range=(0, 1)).fit_transform(z.reshape(-1, 1))
        # z /= 10 ** (math.floor(math.log10(abs(z.max()))))

        bins = 360
        counts, x_hist, y_hist = np.histogram2d(x, y, bins=bins)
        # print('counts shape', counts.shape)
        x, y, z = np.zeros(bins*bins), np.zeros(bins*bins), np.zeros(bins*bins)
        ii = 0
        for xi in range(bins):
            x_ii = (x_hist[xi] + x_hist[xi+1]) / 2.
            for yi in range(bins):
                if counts[xi, yi] > 0:
                    x[ii], y[ii] = x_ii, (y_hist[yi] + y_hist[yi+1]) / 2.
                    z[ii] = counts[
                            max(0, xi-4):min(xi+5, counts.shape[0]), max(0, yi-4):min(yi+5, counts.shape[1])].sum()
                ii += 1
        # print(ii)
        idx = np.where(z > 0)[0]
        x, y, z = x[idx], y[idx], z[idx]

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    # plt.scatter(x, y, c=z, **kwargs)
    ax.scatter(x, y, c=z, **kwargs)
    
    # plt.colorbar(ax=ax)
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    if inline is None:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        # cbar.ax.set_ylabel('Density')
    # elif inline == 'end':
    #     cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, ticks=[])

    return ax


def plot_scatter_density_v1(x, y, fig=None, ax=None, sort=True, bins=100, x_scale=100, y_scale=100, r2=0.8, inline=None, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()

    # x_range = x.max() - x.min()
    # y_range = y.max() - y.min()
    # xyp = x_range / y_range
    # yxp = y_range / x_range
    #
    # r2p = (x_range + y_range) / 70

    x_scaled = (x - x.min()) / (x.max() - x.min()) * x_scale
    y_scaled = (y - y.min()) / (y.max() - y.min()) * y_scale

    z = np.ones(x.size)
    for ii in range(x.size):
        # z[ii] = np.where(yxp * (x[ii] - x) ** 2 + xyp * (y[ii] - y) ** 2 <= r2p)[0].size
        z[ii] = np.where((x_scaled[ii] - x_scaled)**2 + (y_scaled[ii] - y_scaled)**2 <= r2)[0].size

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    if inline is None:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        # cbar.ax.set_ylabel('Density')
    # elif inline == 'end':
    #     cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, ticks=[])

    return ax


def plot_feature_values(data_df, columns_list, correlation_df, target_col, numeric=True, target_type='regression',
                        plot_style='scatterdense', inline=False, inweb=False, set_plot_order=None,
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
    
    hist_params = {"discrete": True, "shrink": 0.6, "multiple": "dodge"}

    if set_plot_order is not None:
        box_params["order"] = set_plot_order
        hist_params["hue_order"] = set_plot_order

    data_df_reset = data_df.reset_index()

    set_ylim = False
    if target_type == 'regression':
        # Check for strong outliers in target column:
        med = data_df_reset[target_col].median()
        std = data_df_reset[target_col].std()
        xx = np.where(data_df_reset[target_col].values > med + 10*std)[0]
        xx = np.append(xx, np.where(data_df_reset[target_col].values < med - 10*std)[0])
        if xx.size > 0:
            print('Target outlier points:', data_df_reset[target_col].values[xx], f'({xx.size})')
            data_df_reset = data_df_reset.drop(xx).reset_index()

            # target_col_vals = data_df.reset_index().drop(xx)[target_col].values
            # target_min, target_max = np.min(target_col_vals), np.max(target_col_vals)
            # max_minus_min = target_max - target_min
            # ymin = target_min - 0.025*max_minus_min
            # ymax = target_max + 0.025*max_minus_min
            # print('New target min/max values:', target_min, target_max)
            # print('Set y-axis limits (for display only): {:.2f} {:.2f}.\n'.format(ymin, ymax))
            # set_ylim = True

    if inline:
        sns.set_theme(style="ticks", font_scale=0.8)
        num_rows = 1 if len(columns_list) <= 3 else 2
        
        if plot_style == 'scatterdense':
            f, axs = plt.subplots(num_rows, 4, figsize=(8.2, 2.8*num_rows), gridspec_kw={'width_ratios': [12, 12, 12, 1]})
            for row in range(num_rows):
                cax = axs[row, 3] if num_rows > 1 else axs[3]
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                cbar = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax, ticks=[])
        else:
            f, axs = plt.subplots(num_rows, 3, figsize=(8.2, 2.8*num_rows))
        
        f.subplots_adjust(wspace=0.15, hspace=0.25)

    else:
        if inweb:
            sns.set_theme(style="ticks", font_scale=1.4)
        else:
            sns.set_theme(style="ticks")
    
    print('Generating plots of {} features...'.format('numeric' if numeric else 'non-numeric/categorical'))
    num_plots = 6 if inline else len(columns_list)
    for jj, column in enumerate(tqdm(columns_list[0:num_plots])):

        if inline:
            num_row = int(np.floor(jj / 3))
            num_col = jj - (num_row * 3)
            ax = axs[num_row, num_col] if num_rows > 1 else axs[num_col]
            inline_scat = 'end' if num_col == 2 else 'inner'
        else:
            f, ax = plt.subplots(figsize=(9, 6))
            inline_scat = None

        data_df_col_notnull = data_df_reset[[column, target_col]].dropna().reset_index()

        # TODO: User can define this value:
        num_uniq = correlation_df.loc[column, "Num Unique Values"]
        if (not numeric) or (num_uniq <= 10):

            num_cat_thresh = 10 if inline else 20
            if num_uniq > num_cat_thresh:
                orig_len = len(data_df_col_notnull)
                value_counts_index = data_df_col_notnull[column].value_counts().index[0:num_cat_thresh]
                data_df_col_notnull = data_df_col_notnull.loc[data_df_col_notnull[column].isin(value_counts_index)]
                print(f"For '{column}', more than {num_cat_thresh} unique values: Only plotting top {num_cat_thresh}, "
                      f"which is {len(data_df_col_notnull)} out of {orig_len} total datapoints.")
                if inline:
                    anc = AnchoredText(f'Plotting {num_cat_thresh} out of {num_uniq} uniq vals', loc="upper left",
                                       pad=0.2, frameon=False, prop={'size': 'small'})
                else:
                    anc = AnchoredText(f'Plotting top {num_cat_thresh} out of {num_uniq} total uniq vals',
                                       loc="upper left", frameon=False)
                ax.add_artist(anc)
            
            if target_type == 'regression':
                # Regression -- Discrete variable

                if not numeric:
                    # Standard Box Plot with X-axis ordered by median value in each category
                    xaxis_order = data_df_col_notnull.groupby(
                        by=[column]).median().sort_values(by=[target_col]).index.tolist()

                    ax = sns.boxplot(data_df_col_notnull, x=column, y=target_col, order=xaxis_order, ax=ax, **box_params)

                else:
                    # Standard Box Plot
                    ax = sns.boxplot(data_df_col_notnull, x=column, y=target_col, ax=ax, **box_params)  # hue="method", palette="vlag"

                # Add in points to show each observation
                if (plot_style != 'scatterdense') and (len(data_df_col_notnull) > 1000):
                    data_df_col_notnull = data_df_col_notnull.sample(n=1000, replace=False)

                if plot_style in ('swarm', 'seaborn'):
                    ax = sns.swarmplot(data_df_col_notnull, x=column, y=target_col, ax=ax, size=2, color=".3", warn_thresh=0.4)

                elif plot_style == 'strip':
                    ax = sns.stripplot(data_df_col_notnull, x=column, y=target_col, ax=ax, jitter=0.25, size=2, color=".3")

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

                    ax = plot_scatter_density(x_all, y_all, fig=f, ax=ax, bins=120, inline=inline_scat, s=3, cmap='viridis')
                
                if (not numeric) and inline and (num_uniq >= 3):
                    xticks_loc, xticks_lab = [], []
                    for cat in ax.get_xticklabels():
                        xticks_loc.append(cat.get_position()[0])
                        xticks_lab.append(cat.get_text()[0:2])
                    ax.set_xticks(xticks_loc, xticks_lab)
            
            else:
                # Classification -- Continuous Variable
                if plot_style == 'seaborn':
                    ax = sns.histplot(data_df_col_notnull, x=column, hue=target_col, ax=ax, **hist_params)  # "stack"
                else:
                    with sns.color_palette('viridis'):
                        ax = sns.histplot(data_df_col_notnull, x=column, hue=target_col, ax=ax, **hist_params)  # "stack"
                ax.set_xticks(data_df_col_notnull[column].unique())

            if (not numeric) and (num_uniq >= 10) and (not inline):
                plt.xticks(rotation=45)
                plt.grid(axis='x')

            if not inline:
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
                # Regression -- Continuous variable
                if plot_style == 'seaborn':
                    ax = sns.scatterplot(data_df_col_notnull, x=column, y=target_col, ax=ax, size=2, legend=False)

                else:
                    ax = plot_scatter_density(data_df_col_notnull[column].values,
                                              data_df_col_notnull[target_col].values, fig=f, ax=ax, bins=120,
                                              inline=inline_scat, s=3, cmap='viridis')

                    # plt.hist2d(data_df_col_notnull[column], data_df_col_notnull[target_col], bins=(100, 100),
                    #            cmap='viridis', cmin=1)  # BuPu
                    # plt.colorbar()

                    # ax.scatter(x, y, c=z, s=100, edgecolor='')
                    # ax.scatter(x, y, c=z, s=50)
            
            else:
                # Classification -- Continuous variable
                ax = sns.boxplot(data_df_col_notnull, x=column, y=target_col, orient='y', ax=ax, **box_params)

                if (plot_style != 'scatterdense') and (len(data_df_col_notnull) > 1000):
                    data_df_col_notnull = data_df_col_notnull.sample(n=1000, replace=False)

                if plot_style in ('swarm', 'seaborn'):
                    ax = sns.swarmplot(data_df_col_notnull, x=column, y=target_col, orient='y', ax=ax, size=2,
                                       color=".3", warn_thresh=0.4)

                elif plot_style == 'strip':
                    ax = sns.stripplot(data_df_col_notnull, x=column, y=target_col, orient='y', jitter=0.25, ax=ax,
                                       size=2, color=".3")
                
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

                    ax = plot_scatter_density(x_all, y_all, fig=f, ax=ax, bins=120, inline=inline_scat, s=3,
                                              cmap='viridis')

            if not inline:
                plt.grid()

            # plt.xlabel(column)
            ax.set_xlabel(column)
        
            if (not inline) or (num_col == 0):
                ax.set_ylabel(target_col)
        
        if inline and (num_col > 0):
            ax.set(ylabel=None)

        if set_ylim:
            plt.ylim(ymin, ymax)

        title_txt = f'{column} vs {target_col} : '
        if numeric and target_type == 'regression':
            title_txt += f'P={correlation_df.loc[column, "Pearson"]}, '
        title_txt += f'MI={correlation_df.loc[column, "Mutual Info"]}, RF={correlation_df.loc[column, "Random Forest"]}'
        if not numeric:
            title_txt += f', RF_norm={correlation_df.loc[column, "RF_norm"]}'
        if not inline:
            ax.set_title(title_txt)

        if ((target_type == 'regression') or (not numeric) or (num_uniq <= 10)) and inline:
            ax.ticklabel_format(axis='y', scilimits=(0, 0))
        if (not inline) and (not inweb):
            plt.savefig('{}/{}_vs_{}.png'.format(plots_folder, column, target_col), bbox_inches='tight')
            plt.close()

    # mpl.use(backend_)  # Reset backend
    # print('*** {} ***'.format(mpl.get_backend()))

    if inline:
        plt.show()

