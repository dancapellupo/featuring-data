
import matplotlib.pyplot as plt
import seaborn as sns


def plot_inline_scatter(data_df, x_col, y_col, title='', overplot=False, outfile=True, plots_folder='./'):

    if not overplot:
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)

    sns.scatterplot(data_df, x=x_col, y=y_col, size=3, legend=False)

    if outfile:
        plt.savefig('{}/{}.png'.format(plots_folder, title), bbox_inches='tight')


def plot_xy(x, y, xlabel=None, ylabel=None, leg_label='', title='', overplot=False, outfile=True, plots_folder='./'):

    if not overplot:
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)

    # sns.lineplot(x=x, y=y, size=3, legend='auto')
    # sns.scatterplot(x=x, y=y, size=3, label=leg_label)
    plt.plot(x, y, 'o', markersize=3, label=leg_label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend()

    if outfile:
        # TODO Take title, lowercase and replace spaces with underscores
        plt.savefig('{}/{}.png'.format(plots_folder, title), bbox_inches='tight')


