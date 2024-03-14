
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns


def plot_feature_values(data_df, columns_list, correlation_df, target_col, numeric=True, catplot_style='swarm',
                        plots_folder='./plots'):

    # backend_ = mpl.get_backend()
    # print('*** {} ***'.format(backend_))
    # mpl.use("Agg")
    # print('*** {} ***'.format(mpl.get_backend()))

    # use_sample = False
    if len(data_df) > 1000:
        data_df_sample = data_df.sample(n=1000, replace=False)
        # use_sample = True

    sns.set_theme(style="ticks")

    for jj, column in enumerate(columns_list):

        f, ax = plt.subplots(figsize=(9, 6))

        # TODO: Use already calculated DF of unique values:
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

