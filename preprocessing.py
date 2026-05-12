"""
Preprocessing class for handling data manipulation and analysis.
"""
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
        self.data_path = dataset_path
        self.data_df = pd.read_csv(self.data_path)
        self.metrics_df = None
        self.non_value_columns_lists = None
        self.p_values_df = None

    @staticmethod
    def apply_filter(input_df, filter_variable, filter_list):
        """
        Applies a filter to the input dataset based on specified unique string indices for a given 
        variable.
        """
        filter_variable_unique_values = \
            sorted(list(set(input_df[filter_variable].tolist())))
        filter_values = [filter_variable_unique_values[i] for i in filter_list]
        input_df = input_df[input_df[filter_variable].isin(filter_values)]
        return input_df

    def get_non_value_columns_lists(self,
                                    column_names):
        """
        Generates lists of unique strings for specified non-value columns in the dataset.
        """
        self.non_value_columns_lists = [None] * len(column_names)
        for u, current_column_name in enumerate(column_names):
            # gather all unique strings in current column
            curr_unique_strings = sorted(list(set(self.data_df[current_column_name].tolist())))
            # append to non-value columns lists
            self.non_value_columns_lists[u] = curr_unique_strings



    def get_difference_from_baseline(self,
                                     subject_variable,
                                     dependent_variable):
        """ 
        Calculates the difference of the dependent variable from baseline for each subject and
        adds it as a new column in the dataframe.
        """
        subject_list = self.data_df[subject_variable].unique().tolist()
        # initialize list of lists to store dependent_variable differences for each subject
        subject_difference_list = [None] * len(subject_list)
        # iterate through subjects and calculate dependent_variable differences from baseline
        # for each subject
        for q, current_subject in enumerate(subject_list):
            current_subject_df = self.data_df[self.data_df[subject_variable] == current_subject]
            current_dependent_variable_array = current_subject_df[dependent_variable].to_numpy()
            current_dependent_variable_difference_list = \
                (current_dependent_variable_array - current_dependent_variable_array[0]).tolist()
            subject_difference_list[q] = current_dependent_variable_difference_list
        # concatenate list of lists into single list
        dependent_variable_difference_list = \
            [item for sublist in subject_difference_list for item in sublist]
        # insert dependent_variable difference list as new column in dataframe
        self.data_df[dependent_variable + '_difference'] = \
            dependent_variable_difference_list

    def get_difference_metrics(self,
                               condition_variable,
                               group_variable,
                               difference_variable,
                               save_csv=False):
        """
        Calculates average, low and high quartiles of the difference variable for each 
        combination of condition and group, and saves the metrics as a new dataframe and 
        optionally as a CSV file. 
        """
        # gather lists of unique strings for condition and group non-value columns
        column_names = [condition_variable, group_variable]
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
        self.metrics_df = \
            pd.DataFrame(condition_metrics_list,
                         columns=['average', 'q1', 'q3', column_names[0], column_names[1]])
        if save_csv:
            self.metrics_df.to_csv('metrics.csv', index=False)

    def get_p_values(self,
                     condition_variable,
                     group_variable,
                     dependent_variable,
                     save_csv=False):
        """
        Calculates p-values for the group variable between groups for each condition 
        using a statistical test, and saves the p-values as a new dataframe and optionally as 
        a CSV file.
        """
        # gather lists of unique strings for condition and group non-value columns
        column_names = [condition_variable, group_variable]
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
                continue
            if w != self.non_value_columns_lists[1][0] and previous_w is None:
                raise ValueError(r'Previous group unique string variable is None.')
            # isolate current condition and group subset of dataframe
            current_subset_df1 = \
                self.data_df[(self.data_df[column_names[0]] == h) &
                             (self.data_df[column_names[1]] == previous_w)]
            current_subset_df2 = \
                self.data_df[(self.data_df[column_names[0]] == h) &
                             (self.data_df[column_names[1]] == w)]
            # gather dependent variable column for current subsets
            current_dependent_variable_array1 = \
                current_subset_df1[dependent_variable].to_numpy()
            current_dependent_variable_array2 = \
                current_subset_df2[dependent_variable].to_numpy()
            # calculate p_value for current subsets and store in list
            _, p_value = mannwhitneyu(current_dependent_variable_array1,
                                      current_dependent_variable_array2)
            current_p_value_list = [p_value, h, w]
            condition_p_value_list.append(current_p_value_list)
            # update previous group unique string variable
            previous_w = w
        # convert list of lists to dataframe and save as csv
        self.p_values_df = \
            pd.DataFrame(condition_p_value_list,
                         columns=['p_value', column_names[0], column_names[1]])
        if save_csv:
            self.p_values_df.to_csv('p_values.csv', index=False)

    def get_preprocessed_data(self,
                              subject_variable,
                              dependent_variable,
                              condition_variable,
                              group_variable,
                              save_csv_files=False):
        """
        Main method to execute the preprocessing steps, including optional filtering, 
        calculating metrics, and generating p-values.
        """
        # gather measure differences from baseline
        self.get_difference_from_baseline(subject_variable,
                                          dependent_variable)
        # gather average, low and high quartiles for each group
        difference_variable = self.data_df.columns[-1]
        self.get_difference_metrics(condition_variable,
                                    group_variable,
                                    difference_variable,
                                    save_csv=save_csv_files)
        # gather p-values for each group
        self.get_p_values(condition_variable,
                          group_variable,
                          dependent_variable,
                          save_csv=save_csv_files)


if __name__ == '__main__':

    # --- read data ---
    EXAMPLE_DATA_PATH = r'.\chickweight.csv'
    example_preprocessor = Preprocessor(EXAMPLE_DATA_PATH)

    # --- specify variables ---
    example_subject_variable = example_preprocessor.data_df.columns[2]
    example_dependent_variable = example_preprocessor.data_df.columns[0]
    example_condition_variable = example_preprocessor.data_df.columns[3]
    example_group_variable = example_preprocessor.data_df.columns[1]

    # --- gather preprocessed data ---
    example_preprocessor.get_preprocessed_data(example_subject_variable,
                                               example_dependent_variable,
                                               example_condition_variable,
                                               example_group_variable,
                                               save_csv_files=True)
