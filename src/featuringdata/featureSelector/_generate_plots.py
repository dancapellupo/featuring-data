
import matplotlib.pyplot as plt
import seaborn as sns


def convert_title_to_filename(title):
    return title.lower().replace(' ', '_')


def plot_inline_scatter(data_df, x_col, y_col, f=None, ax=None, xlabel=None, ylabel=None, title='', overplot=False,
                        leg_label='', hline=None, vline=None, reverse_x=False, outfile=True, plots_folder='./'):

    if not overplot:
        sns.set_theme(style="ticks", font_scale=1.2)
        f, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)

    # sns.scatterplot(data_df, x=x_col, y=y_col, size=3, legend=False)
    plt.plot(data_df[x_col], data_df[y_col], 'o', markersize=6, label=leg_label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if (hline is not None) or (vline is not None):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if hline is not None:
            ax.hlines(hline, xmin, xmax, color='black', linestyles=':')

        if vline is not None:
            ax.vlines(vline, ymin, ymax, color='black', linestyles=':')

        ax.set_xlim(xmax, xmin) if reverse_x else ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if outfile:
        if leg_label != '':
            plt.legend()
        plt.grid()
        plt.savefig('{}/{}.png'.format(plots_folder, title), bbox_inches='tight')
        # *** Close for now ***
        # plt.close(f)
    else:
        return f, ax


def plot_xy(x, y, f=None, ax=None, xlabel=None, ylabel=None, title='', reverse_x=False, overplot=False, outfile=True,
            plots_folder='./', **kwargs):

    if not overplot:
        sns.set_theme(style="ticks", font_scale=1.1)
        f, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)

    # sns.lineplot(x=x, y=y, size=3, legend='auto')
    # sns.scatterplot(x=x, y=y, size=3, label=leg_label)
    plt.plot(x, y, 'o', **kwargs)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend()

    if outfile:
        if reverse_x:
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmax, xmin)
        plt.grid()
        # DONE Take title, lowercase and replace spaces with underscores
        plt.savefig('{}/{}.png'.format(plots_folder, convert_title_to_filename(title)), bbox_inches='tight')
        plt.close(f)
    else:
        return f, ax


def plot_horizontal_line(ax, y_loc):
    ax.hlines(y_loc, 0, 1, transform=ax.get_yaxis_transform(), color='black', linestyles=':')
    return ax


def plot_vertical_line(ax, x_loc):
    print(ax.get_ylim())
    ax.vlines(x_loc, 0.5, 1, transform=ax.get_xaxis_transform(), color='black', linestyles=':')
    print(ax.get_ylim())
    return ax


def save_fig(f, ax, plots_folder='./', title=''):
    plt.savefig('{}/{}.png'.format(plots_folder, title), bbox_inches='tight')
    plt.close(f)


def plot_xy_splitaxis(x, y, title='test', plots_folder='./'):

    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.subplots_adjust(wspace=0.05)  # adjust space between axes

    # plot the same data on both axes
    ax1.plot(x, y, 'o', markersize=3)
    ax2.plot(x, y, 'o', markersize=3)

    # zoom-in / limit the view to different portions of the data
    ax1.set_xlim(-1, 140)  # outliers only
    ax2.set_xlim(265, 300)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.savefig('{}/{}.png'.format(plots_folder, title), bbox_inches='tight')
    plt.close()




