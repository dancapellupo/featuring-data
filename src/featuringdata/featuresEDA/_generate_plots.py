
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns


def plot_feature_values(data_df, columns_list, correlation_df, target_col, numeric=True, catplot_style='swarm',
                        plots_folder='./plots'):

    backend_ = mpl.get_backend()
    print('*** {} ***'.format(backend_))
    mpl.use("Agg")
    print('*** {} ***'.format(mpl.get_backend()))

    sns.set_theme(style="ticks")

    for jj, column in enumerate(columns_list):

        f, ax = plt.subplots(figsize=(9, 6))

        # TODO: Use already calculated DF of unique values:
        if (not numeric) or (np.unique(data_df[column]).size <= 10):
            # Plot the orbital period with horizontal boxes
            sns.boxplot(
                data_df, x=column, y=target_col,  # hue="method",
                whis=[0, 100], width=.6, # palette="vlag"
            )

            # Add in points to show each observation
            if catplot_style == 'swarm':
                sns.swarmplot(data_df, x=column, y=target_col, size=2, color=".3")
            else:
                sns.stripplot(data_df, x=column, y=target_col, jitter=0.25, size=2, color=".3")

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

        # TODO: Create directory based on time-stamp
        plt.savefig('{}/{}_vs_{}.png'.format(plots_folder, column, target_col), bbox_inches='tight')

    mpl.use(backend_)  # Reset backend
    print('*** {} ***'.format(mpl.get_backend()))

