"""
Preprocessing class for handling data manipulation and analysis.
"""
import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu


class Preprocessor:
    """
    This class provides methods for preprocessing a dataset, including filtering, calculating 
    differences from baseline, computing metrics, and performing statistical tests to obtain 
    p-values. The preprocessed data can be saved as CSV files for further analysis or 
    visualization.
    """

    def __init__(self, dataset_path):
        self.data_df = pd.read_csv(dataset_path)
        self.prep_variables = {
            "subject": self.data_df.columns[2],
            "dependent": self.data_df.columns[0],
            "condition": self.data_df.columns[3],
            "group": self.data_df.columns[1]
        }
        self.non_value_columns_lists = None
        self.prep_results = {
            "metrics": pd.DataFrame(),
            "metrics variables": None,
            "p-values": pd.DataFrame(),
            "p-values variable": None,
        }

    @staticmethod
    def apply_filter(input_df, filter_variable, filter_list):
        """
        Applies a filter to the input dataset based on specified unique string indices for a given 
        variable.
        """
        filter_variable_unique_values = \
            sorted(list(set(input_df[filter_variable].tolist())))
        filter_values = [filter_variable_unique_values[i] for i in filter_list]
        input_df = input_df[input_df[filter_variable].isin(filter_values)].copy()
        return input_df

    def apply_multiple_filters(self, condition_filters, group_filters):
        """ 
        Applies multiple filters to the preprocessing results dataframes based on specified unique 
        string indices for condition and group variables.
        """
        # assign metrics and p-values dataframes keys
        prep_results_keys = [list(self.prep_results)[0], list(self.prep_results)[2]]
        # if condition filters specified, apply to preprocessing results dataframes
        if condition_filters:
            for current_key in prep_results_keys:
                self.prep_results[current_key] = \
                    self.apply_filter(self.prep_results[current_key],
                                      self.prep_variables["condition"],
                                      condition_filters)
        # if group filters specified, apply to preprocessing results dataframes
        if group_filters:
            for current_key in prep_results_keys:
                self.prep_results[current_key] = \
                    self.apply_filter(self.prep_results[current_key],
                                      self.prep_variables["group"],
                                      group_filters)

    def get_x_tick_labels(self):
        """ 
        Obtains x tick labels for the line plot based on the unique values of the group variable,
        with special handling for a baseline value of 0.
        """
        group_unique_values = \
           sorted(list(set(self.prep_results["metrics"][self.prep_variables["group"]].tolist())))
        x_tick_labels = None
        if 0 in group_unique_values:
            x_tick_labels = \
                ['hatching'] + [str(s) + " days" for s in group_unique_values[1:]]
        elif 0 not in group_unique_values:
            x_tick_labels = [str(s) + " days" for s in group_unique_values]
        if x_tick_labels is None:
            raise ValueError(r'The x tick labels list is None.')
        return x_tick_labels

    def get_non_value_columns_lists(self, column_names):
        """
        Generates lists of unique strings for specified non-value columns in the dataset.
        """
        self.non_value_columns_lists = [None] * len(column_names)
        for u, current_column_name in enumerate(column_names):
            # gather all unique strings in current column
            curr_unique_strings = sorted(list(set(self.data_df[current_column_name].tolist())))
            # append to non-value columns lists
            self.non_value_columns_lists[u] = curr_unique_strings

    def get_difference_from_baseline(self):
        """ 
        Calculates the difference of the dependent variable from baseline for each subject and
        adds it as a new column in the dataframe.
        """
        subject_list = \
            self.data_df[self.prep_variables["subject"]].unique().tolist()
        # initialize list of lists to store dependent variable differences for each subject
        subject_difference_list = [None] * len(subject_list)
        # iterate through subjects and calculate dependent variable differences from baseline
        # for each subject
        for q, current_subject in enumerate(subject_list):
            current_subject_df = \
                self.data_df[self.data_df[self.prep_variables["subject"]] == \
                    current_subject]
            current_dependent_variable_array = \
                current_subject_df[self.prep_variables["dependent"]].to_numpy()
            current_dependent_variable_difference_list = \
                (current_dependent_variable_array - current_dependent_variable_array[0]).tolist()
            subject_difference_list[q] = current_dependent_variable_difference_list
        # concatenate list of lists into single list
        dependent_variable_difference_list = \
            [item for sublist in subject_difference_list for item in sublist]
        # insert dependent_variable difference list as new column in dataframe
        self.data_df[self.prep_variables["dependent"] + '_difference'] = \
            dependent_variable_difference_list

    def get_difference_metrics(self, difference_variable, save_csv=False):
        """
        Calculates average, low and high quartiles of the difference variable for each 
        combination of condition and group, and saves the metrics as a new dataframe and 
        optionally as a CSV file. 
        """
        # gather lists of unique strings for condition and group non-value columns
        column_names = [self.prep_variables["condition"],
                        self.prep_variables["group"]]
        if not self.non_value_columns_lists:
            self.get_non_value_columns_lists(column_names)
        # initialize list of lists to store metrics for each condition and group
        condition_metrics_list = []
        # iterate through all combinations of condition and group and store metrics
        # '*' operator unpacks the list iterable
        for h, w in itertools.product(*self.non_value_columns_lists):
            # isolate current condition and group subset of dataframe
            current_subset_df = self.data_df[(self.data_df[column_names[0]] == h) &
                                            (self.data_df[column_names[1]] == w)]
            # gather measure difference column for current subset
            current_difference_array = \
                current_subset_df[difference_variable].to_numpy()
            # calculate average, low and high quartiles for current subset and store in list
            current_metrics_list = \
                [np.mean(current_difference_array),
                 np.quantile(current_difference_array, 0.25),
                 np.quantile(current_difference_array, 0.75),
                 h, w]
            condition_metrics_list.append(current_metrics_list)
        # convert list of lists to dataframe and save as csv
        self.prep_results["metrics"] = \
            pd.DataFrame(condition_metrics_list,
                         columns=['average', 'q1', 'q3', column_names[0], column_names[1]])
        self.prep_results["metrics variables"] = \
            [self.prep_results["metrics"].columns[i] for i in range(3)]
        if save_csv:
            self.prep_results["metrics"].to_csv('metrics.csv', index=False)

    def get_p_values(self, save_csv=False):
        """
        Calculates p-values for the group variable between groups for each condition 
        using a statistical test, and saves the p-values as a new dataframe and optionally as 
        a CSV file.
        """
        # gather lists of unique strings for condition and group non-value columns
        column_names = [self.prep_variables["condition"],
                        self.prep_variables["group"]]
        if not self.non_value_columns_lists:
            self.get_non_value_columns_lists(column_names)
        # initialize previous group unique string variable
        previous_w = None
        # initialize list of lists to store p-values for each condition and group
        condition_p_value_list = []
        # iterate through all combinations of condition and group and store p-values
        # '*' operator unpacks the list iterable
        for h, w in itertools.product(*self.non_value_columns_lists):
            # exclude baseline group unique string
            if w == self.non_value_columns_lists[1][0]:
                previous_w = w
                current_p_value_list = [None, h, w]
                condition_p_value_list.append(current_p_value_list)
                continue
            if w != self.non_value_columns_lists[1][0] and previous_w is None:
                raise ValueError(r'Previous group unique string variable is None.')
            # isolate current condition and group subset of dataframe
            current_sub_df1 = \
                self.data_df[(self.data_df[column_names[0]] == h) &
                             (self.data_df[column_names[1]] == previous_w)]
            current_sub_df2 = \
                self.data_df[(self.data_df[column_names[0]] == h) &
                             (self.data_df[column_names[1]] == w)]
            # calculate p_value for current subsets and store in list
            _, p_value = \
                mannwhitneyu(current_sub_df1[self.prep_variables["dependent"]].to_numpy(),
                             current_sub_df2[self.prep_variables["dependent"]].to_numpy())
            current_p_value_list = [p_value, h, w]
            condition_p_value_list.append(current_p_value_list)
            # update previous group unique string variable
            previous_w = w
        # convert list of lists to dataframe and save as csv
        self.prep_results["p-values"] = \
            pd.DataFrame(condition_p_value_list,
                         columns=['p_value', column_names[0], column_names[1]])
        self.prep_results["p-values variable"] = \
            self.prep_results["p-values"].columns[0]
        if save_csv:
            self.prep_results["p-values"].to_csv('p_values.csv', index=False)

    def get_preprocessed_data(self, save_csv_files=False):
        """
        Main method to execute the preprocessing steps, including optional filtering, 
        calculating metrics, and generating p-values.
        """
        if not os.path.exists(r'.\metrics.csv') and not os.path.exists(r'.\p_values.csv'):
            # gather measure differences from baseline
            self.get_difference_from_baseline()
            # gather average, low and high quartiles for each group
            difference_variable = self.data_df.columns[-1]
            self.get_difference_metrics(difference_variable, save_csv=save_csv_files)
            # gather p-values for each group
            self.get_p_values(save_csv=save_csv_files)
        elif os.path.exists(r'.\metrics.csv') and os.path.exists(r'.\p_values.csv'):
            self.prep_results["metrics"] = pd.read_csv(r'.\metrics.csv')
            self.prep_results["metrics variables"] = \
                [self.prep_results["metrics"].columns[i] for i in range(3)]
            self.prep_results["p-values"] = pd.read_csv(r'.\p_values.csv')
            self.prep_results["p-values variable"] = \
                self.prep_results["p-values"].columns[0]


if __name__ == '__main__':

    # --- read data ---
    EXAMPLE_DATA_PATH = r'.\chickweight.csv'
    example_data = Preprocessor(EXAMPLE_DATA_PATH)
    example_data.get_preprocessed_data(save_csv_files=True)
