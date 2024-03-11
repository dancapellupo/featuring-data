
import matplotlib.pyplot as plt
import seaborn as sns


def plot_inline_scatter(data_df, x_col, y_col, title='', overplot=False, outfile=True, plots_folder='./'):

    if not overplot:
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)

    sns.scatterplot(data_df, x=x_col, y=y_col, size=3, legend=False)

    if outfile:
        plt.savefig('{}/{}_vs_{}.png'.format(plots_folder, x_col, y_col), bbox_inches='tight')

