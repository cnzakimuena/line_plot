"""
This code generates a line plot with error bars. The plot uses the chick weight dataset for 
demonstration, and includes example annotations for significance based on statistical testing. 
The appearance of the plot is customized and the final figure is saved.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pypalettes

from preprocessing import Preprocessor

# non-italic matplotlib greek characters
matplotlib.rcParams['mathtext.default'] = 'regular'


def get_offset_locations(initial_x, count, interval):
    """
    Calculates offsets for a given number of locations.
    """
    if count <= 0:
        return []
    # calculate the starting point relative to the center
    start_offset = -((count - 1) / 2.0) * interval
    # iterate based on count
    locations = []
    for i in range(count):
        current_offset = start_offset + (i * interval)
        locations.append(initial_x + current_offset)
    return locations


def multiples(value, length):
    """
    Generates a list of multiples for a given value and length.
    """
    multiples_list = [*range(value, length*value+1, value)]
    multiples_list.insert(0, 0)
    return multiples_list[1:]


def get_star_number(p_value):
    """
    Determines the number of stars for a given p-value based on significance thresholds.
    """
    star_integer = None
    if 0.01 < p_value <= 0.05:
        star_integer = 1
    elif 0.001 < p_value <= 0.01:
        star_integer = 2
    elif p_value <= 0.001:
        star_integer = 3
    if star_integer is None:
        raise ValueError(r'Star integer is None.')
    return star_integer


def generate_plot(metrics_df,
                  group_variable,
                  metrics_variable_list,
                  condition_variable,
                  y_range,
                  **plot_kwargs):
    """
    Generates a line plot with error bars and optional p-value annotations 
    based on the provided dataframes and parameters.
    """

    # default color palette assignment if not provided
    condition_unique_values = sorted(list(set(metrics_df[condition_variable].tolist())))
    if 'palette_list' not in plot_kwargs:
        plot_kwargs['palette_list'] = sns.color_palette("Blues", len(condition_unique_values))

    # assign initial independent variable values
    group_unique_values = sorted(list(set(metrics_df[group_variable].tolist())))
    global_x = multiples(2, len(group_unique_values))
    # gather independent variable offset values
    count = len(condition_unique_values)
    x_lists = [([None] * len(global_x)) for _ in range(count)]
    for q, x_element in enumerate(global_x):
        curr_offset_locations = get_offset_locations(x_element, count, 1)
        for w, _ in enumerate(condition_unique_values):
            x_lists[w][q] = curr_offset_locations[w]

    # assign reverse zorder list
    zorder_list = list(range(len(condition_unique_values)))[::-1]

    # set font and font size
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams.update({'font.size': 22})

    # generate plot
    fig, ax = plt.subplots(figsize=(10, 9))

    x_range = [global_x[0] - 1.5, global_x[-1] + 1.5]
    plt.xlim(x_range)
    plt.ylim(y_range)

    # iterate trhough conditions
    for q, condition_value in enumerate(condition_unique_values):
        # current color
        curr_color_name = plot_kwargs['palette_list'][q]
        # current metrics values
        subset_data_df = metrics_df[metrics_df[condition_variable] == condition_value]
        y = subset_data_df[metrics_variable_list[0]].tolist()
        yerr = \
            np.array([(subset_data_df[metrics_variable_list[0]] -
                    subset_data_df[metrics_variable_list[1]]).to_numpy(),
                    (subset_data_df[metrics_variable_list[2]] -
                    subset_data_df[metrics_variable_list[0]]).to_numpy()])
        # current group values
        x = x_lists[q]
        # draw line plot
        (_, caps, _) = plt.errorbar(x, y, yerr, color=curr_color_name,
                                    solid_capstyle='projecting', capsize=12, linewidth=4,
                                    zorder=zorder_list[q])
        for cap in caps:
            cap.set_markeredgewidth(4)
        plt.scatter(x=x, y=y, c=curr_color_name, s=140, zorder=zorder_list[q])
        # if p-values are given, add to line plot
        if 'p_values_df' in plot_kwargs:
            subset_p_values_df = \
                plot_kwargs['p_values_df'][plot_kwargs['p_values_df'][condition_variable] ==
                                           condition_value]

            # add p-value star above Q3
            current_p_values_list = subset_p_values_df[plot_kwargs['p_values_variable']].tolist()
            current_q3_list = subset_data_df[metrics_variable_list[2]].tolist()
            p_location_reference = (y_range[1]-y_range[0]) * 0.08
            for h, current_p_value in enumerate(current_p_values_list):
                if current_p_value is not None and current_p_value < 0.05:
                    y_location = current_q3_list[h] + p_location_reference
                    star_number = get_star_number(current_p_value)
                    star_string = r'$\mathbf{\ast}$' * star_number
                    ax.annotate(star_string, xy=(x[h], y_location),
                                color=curr_color_name,
                                fontsize="xx-small", weight='normal',
                                horizontalalignment='center',
                                verticalalignment='center')

    plt.yticks(np.linspace(y_range[0], y_range[1], num=5).tolist())
    if 'specified_x_tick_labels' in plot_kwargs:
        plt.xticks(global_x, plot_kwargs['specified_x_tick_labels'])
        ax.tick_params(axis='x', rotation=45)
    if 'specified_x_label' in plot_kwargs and 'specified_x_label_text_padding' in plot_kwargs:
        plt.xlabel(plot_kwargs['specified_x_label'],
                   labelpad=plot_kwargs['specified_x_label_text_padding'])
    if 'specified_y_label' in plot_kwargs and 'specified_y_label_text_padding' in plot_kwargs:
        plt.ylabel(plot_kwargs['specified_y_label'],
                   labelpad=plot_kwargs['specified_y_label_text_padding'])
    ax.tick_params(axis='both', colors='black')

    # set the color of the axis labels
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # change spines
    for axis in ['left', 'bottom']:
        plt.gca().spines[axis].set_linewidth(3)
    plt.gca().tick_params(width=3)

    # adjust subplots spacing
    # if subplots are added, can include, for e.g., 'wspace=0.4, hspace=0.4'
    # to control padding between subplots
    plt.subplots_adjust(bottom=0.35, top=0.85, left=0.2, right=0.85)

    # add global title
    if 'super_title' in plot_kwargs:
        fig.suptitle(plot_kwargs['super_title'], fontsize="large", color="black")

    # add legend
    condition_names = [condition_variable + " " + str(i) for i in condition_unique_values]
    colors = dict(zip(condition_names, plot_kwargs['palette_list']))
    labels = list(colors.keys())
    circle_handles = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=colors[label], markersize=16) for label in labels]
    plt.legend(circle_handles, labels, frameon=False, bbox_to_anchor=(0.4, 0.98), ncol=1)

    # draw lines below subplots
    trans = ax.get_xaxis_transform()
    ax.plot([3.5, 8.5], [-.27, -.27], color="black", transform=trans, clip_on=False, linewidth=3)
    plt.figtext(0.45, 0.19, r"IGF1R $\mathbf{\uparrow}$",
                ha="center", va="top", fontsize=24, color="black")
    ax.plot([9.5, 14.5], [-.27, -.27], color="black", transform=trans, clip_on=False, linewidth=3)
    plt.figtext(0.7, 0.19, r"IGF1R $\mathbf{\downarrow}$",
                ha="center", va="top", fontsize=24, color="black")


if __name__ == '__main__':

    # --- read data ---
    EXAMPLE_DATA_PATH = r'.\chickweight.csv'
    example_data = Preprocessor(EXAMPLE_DATA_PATH)
    example_data.get_preprocessed_data(save_csv_files=False)
    # apply filters (optional)
    example_data.apply_multiple_filters([0, 2], list(range(0, 7)))
    # specify data-related plotting parameters (optional)
    example_x_tick_labels = example_data.get_x_tick_labels()

    # palette setup (optional)
    example_condition_list = \
        example_data.prep_results["metrics"][example_data.prep_variables["condition"]].tolist()
    example_condition_unique_values = sorted(list(set(example_condition_list)))
    cmap = pypalettes.load_cmap("Chlorurus_microrhinos",
                                keep_first_n=len(example_condition_unique_values))
    pypalettes_list = cmap.colors # return colors as a list of hexadecimal values

    # --- plot data ---
    generate_plot(example_data.prep_results["metrics"],
                  example_data.prep_variables["group"],
                  example_data.prep_results["metrics variables"],
                  example_data.prep_variables["condition"],
                  [-50, 150],
                  p_values_df=example_data.prep_results["p-values"],
                  p_values_variable=example_data.prep_results["p-values variable"],
                  specified_x_label='Age',
                  specified_y_label='Weight change [g]',
                  specified_x_label_text_padding=60,
                  specified_y_label_text_padding=10,
                  specified_x_tick_labels=example_x_tick_labels,
                  palette_list=pypalettes_list)

    # save figure
    FILE_DESTINATION = r'.\figure'
    plt.savefig(os.path.join(FILE_DESTINATION + '.pdf').replace("\\", "/"), format="pdf")
    plt.savefig(os.path.join(FILE_DESTINATION + '.png').replace("\\", "/"), dpi=300)
    plt.close()
