

from .logger import LoggingHandler 
from .splits import SlidingWindowTimeSeriesSplit, ExpandingWindowTimeSeriesSplit
from .errors import RawResiduals

import scipy.io as sio
import pandas as pd
import xarray as xr
import numpy as np
import copy
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import KFold

    
    
def load_dummy_data_df(series_count = 10, timestamp_count = 5, time_feature_count = 3, series_feature_count = 2, vs_times_series_factor = 10000, vs_times_timestamps_factor = 100, vs_series_series_factor = 10000):
    colname_series = 'series'
    colname_timestamp = 'timestamp'
    colnames_time_features = [('time_label_' + str(i)) for i in range(1,time_feature_count+1)]
    colnames_series_features = [('series_label_' + str(i)) for i in range(1,series_feature_count+1)]
    
    ts = 1+np.arange(timestamp_count).reshape(-1,+1)
    ft = 1+np.arange(time_feature_count).reshape(+1,-1)
    temp_right_df = pd.DataFrame(columns = ([colname_timestamp] + colnames_time_features), data = np.concatenate((ts, (ts * vs_times_timestamps_factor) + ft), axis=1))
    temp_right_df['key'] = 999
    temp_left_df = pd.DataFrame(columns = [colname_series], data = (1+np.arange(series_count)))
    temp_left_df['key'] = 999
    temp_df = temp_left_df.merge(temp_right_df, how='outer', on='key')
    temp_df = temp_df.drop('key', axis=1)
    temp_df[colnames_time_features] = (temp_df[colname_series].values * vs_times_series_factor).reshape(-1,+1) + temp_df[colnames_time_features]
    dummy_vs_times_df = temp_df
                                
    s = 1+np.arange(series_count).reshape(-1,+1)
    fs = 1+np.arange(series_feature_count).reshape(+1,-1)
    dummy_vs_series_df = pd.DataFrame(columns = ([colname_series] + colnames_series_features), data = np.concatenate((s, (s * vs_series_series_factor) + fs), axis=1))
    
    return (dummy_vs_times_df, dummy_vs_series_df) 

    
# Design patterns used: Flyweight, Prototype.
class MultiSeries(LoggingHandler):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """       
    
    # Pythonic way of doing read-only properties
    @property
    def count_features(self):
        count = 0
        if self._value_colnames_vs_times is not None:
            count = count + len(self._value_colnames_vs_times)
        if self._value_colnames_vs_series is not None:
            count = count + len(self._value_colnames_vs_series)
        return count
        
    # Pythonic way of doing read-only properties
    @property
    def all_non_timestamp_feature_names(self):
        res = self._value_colnames_vs_times
        if self._value_colnames_vs_series is not None:
            res = res + self._value_colnames_vs_series
        return res

    # Pythonic way of doing read-only properties
    @property
    def count_observations(self):
        return self._select_df_obs_vs_times().shape[0]
        
    def _inferValueColnames(self, data_df, time_colname, series_id_colnames, value_colnames, description, check_presence_of_time_colname=True):
        if data_df is None:
            return None
        else:
            all_colnames = list(data_df.columns.values)
            if check_presence_of_time_colname and time_colname not in all_colnames:
                raise Exception('time_colname ' + str(time_colname) + ' is not a column of the given data_df')    
            for c in series_id_colnames:
                if c not in all_colnames:
                    raise Exception('series_id_colnames item ' + str(c) + ' is not a column of the given data_df')
            given_colnames = [time_colname] + series_id_colnames
            if value_colnames is None:
                self.debug('value_colnames was not specified, so will infer from the data frame.')
                other_colnames = list(np.setdiff1d(all_colnames, given_colnames))
                value_colnames = other_colnames
                self.info('Inferred ' + description +' value colnames = ' + str(value_colnames))
            else:
                given_colnames = given_colnames + value_colnames
                other_colnames = list(np.setdiff1d(all_colnames, given_colnames))
                self.info('The following col names will be dropped: ' + str(other_colnames))
            return value_colnames
            
    def __init__(self, time_colname, series_id_colnames, data_vs_times_df, data_vs_series_df=None, value_colnames_vs_times=None, value_colnames_vs_series=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        super(MultiSeries, self).__init__()
        
        # Validation...
        if type(data_vs_times_df) != pd.DataFrame:
            raise Exception('data_vs_times_df must be a pandas DataFrame! Was ' + str(type(data_vs_times_df)) + ' instead.')
        if not(data_vs_series_df is None) and type(data_vs_series_df) != pd.DataFrame:
            raise Exception('data_vs_series_df must be a pandas DataFrame! Was ' + str(type(data_vs_series_df)) + ' instead.')
        if type(time_colname) != str:
            raise Exception('time_colname must be a string! Was ' + str(type(time_colname)) + ' instead.')
        
        str_shape_data_vs_times_df = 'None'
        if data_vs_times_df is not None:
            str_shape_data_vs_times_df = str(data_vs_times_df.shape)
        str_shape_data_vs_series_df = 'None'
        if data_vs_series_df is not None:
            str_shape_data_vs_series_df = str(data_vs_series_df.shape)
        self.info('Initialising MultiSeries: data_vs_times_df.shape = ' + str_shape_data_vs_times_df + ', data_vs_times_df.shape = ' + str_shape_data_vs_series_df + ', time_colname = ' + str(time_colname) + ', series_id_colnames = ' + str(series_id_colnames) + ', value_colnames_vs_times = ' + str(value_colnames_vs_times) + ', value_colnames_vs_series = ' + str(value_colnames_vs_series))
        
        # ... continued
        if type(series_id_colnames) == str:
            series_id_colnames = [series_id_colnames] # for convenience
        if type(value_colnames_vs_times) == str:
            value_colnames_vs_times = [value_colnames_vs_times] # for convenience
        if type(value_colnames_vs_series) == str:
            value_colnames_vs_series = [value_colnames_vs_series] # for convenience
                        
        self._time_colname = time_colname
        self._series_id_colnames = series_id_colnames
        self._value_colnames_vs_times = self._inferValueColnames(data_df=data_vs_times_df, value_colnames=value_colnames_vs_times, description='time-label', time_colname=time_colname, series_id_colnames=series_id_colnames)
        self._value_colnames_vs_series = self._inferValueColnames(data_df=data_vs_series_df, value_colnames=value_colnames_vs_series, description='series-label', time_colname=time_colname, series_id_colnames=series_id_colnames, check_presence_of_time_colname=False)
        
        # Define a mapping between multiple series_id_colnames and a single identifier column
        self._series_id_colname = '***internal_series_id***' # don't clash with anything
        if data_vs_times_df.columns.contains(self._series_id_colname) or ((data_vs_series_df is not None) and data_vs_series_df.columns.contains(self._series_id_colname)):
            raise Exception('Special column ' + self._series_id_colname + ' already exists!')
        self._series_id_colnames_mapping = data_vs_times_df[series_id_colnames].drop_duplicates().reset_index(drop=True)
        if not(data_vs_series_df is None):
            part_two = data_vs_series_df[series_id_colnames].drop_duplicates().reset_index(drop=True)
            self._series_id_colnames_mapping = pd.concat([self._series_id_colnames_mapping, part_two], ignore_index=True)
            self._series_id_colnames_mapping = self._series_id_colnames_mapping.drop_duplicates().reset_index(drop=True)
        self._series_id_colnames_mapping[self._series_id_colname] = self._series_id_colnames_mapping.index.values
            
        # Replace multiple series_id_colnames with a single idenfier column, for the time-label DF
        self._data_vs_times_df = data_vs_times_df[[time_colname] + series_id_colnames + self._value_colnames_vs_times].copy() # take a copy before we start modifying it
        self._data_vs_times_df = self._data_vs_times_df.merge(self._series_id_colnames_mapping)
        self._data_vs_times_df.drop(axis=1, inplace=True, labels=self._series_id_colnames)
        
        # Replace multiple series_id_colnames with a single idenfier column, for the series-label DF
        if data_vs_series_df is None:
            self._data_vs_series_df = None
        else:
            self._data_vs_series_df = data_vs_series_df[series_id_colnames + self._value_colnames_vs_series].copy() # take a copy before we start modifying it
            self._data_vs_series_df = self._data_vs_series_df.merge(self._series_id_colnames_mapping)
            self._data_vs_series_df.drop(axis=1, inplace=True, labels=self._series_id_colnames)
        
        # After merging by columns
        self._series_id_colnames_mapping.set_index(inplace=True, keys=self._series_id_colname)
            
        # Validation of DF contents, for the time-label DF only
        counts_by_time_and_series = self._data_vs_times_df.groupby([self._time_colname,self._series_id_colname]).size()
        counts_by_time = self._data_vs_times_df.groupby([self._time_colname]).size()
        counts_by_series = self._data_vs_times_df.groupby(self._series_id_colname).size()
        self._count_time_indices = counts_by_time.size
        self._count_series_indices = counts_by_series.size
        duplicate_observations = counts_by_time_and_series[counts_by_time_and_series > 1]
        if duplicate_observations.size > 0:
            raise Exception('There are ' + str(duplicate_observations.size) + ' instances of more than one observation per series index + time index! Should be 0 instances.')
        missing_observations = counts_by_time[counts_by_time < self._count_series_indices].size
        if missing_observations > 0:
            self.warning(str(missing_observations) + ' time indices have missing series observations')
        missing_observations = counts_by_series[counts_by_series < self._count_time_indices].size
        if missing_observations > 0:
            self.warning(str(missing_observations) + ' series indices have missing time observations')
                        
        # Sort by single series identifier (for reproducibility) and time (to ensure CV works!)
        # MultiIndex requires data to be sorted to work properly (source: http://pandas.pydata.org/pandas-docs/version/0.18.1/advanced.html)
        self._data_vs_times_df.sort_values(axis=0, inplace=True, by=[self._series_id_colname, self._time_colname])
        if self._data_vs_series_df is not None:
            self._data_vs_series_df.sort_values(axis=0, inplace=True, by=[self._series_id_colname])
        
        # I've never had cause to use this, so have left it commented for now.
        # Drop any NA observations that are being passed in, now that we have extracted series & timestamp info. from them
        #self._data_vs_times_df = self._data_vs_times_df.dropna().reset_index(drop=True)
        #if self._data_vs_series_df is not None:
        #    self._data_vs_series_df = self._data_vs_series_df.dropna().reset_index(drop=True)
        
        # Set an index consisting of series identifier and time, for the time-label DF...
        self._data_vs_times_df.set_index(inplace=True, verify_integrity=False, keys=[self._series_id_colname, self._time_colname]) 
        
        # ... and by series identifier only, for the series-label DF
        if self._data_vs_series_df is not None:
            self._data_vs_series_df.set_index(inplace=True, verify_integrity=False, keys=[self._series_id_colname]) 
            
        # Prepare an index to filter the rows we care about, for the time-label DF
        self._filter_obs = self._data_vs_times_df.index.copy()
        self._series_id_uniques = self._filter_obs.levels[self._filter_obs.names.index(self._series_id_colname)].values # cached here for efficiency
        self._time_uniques_all = self._filter_obs.levels[self._filter_obs.names.index(self._time_colname)].values           # cached here for efficiency
        self._set_derived_filters()
    
        
    def _set_derived_filters(self):
        # This filter is for all possible combinations of values. Any non-observed values will appear as np.nan:s when it is used to index the data DF:
        self._filter_grid = pd.MultiIndex.from_product([self._series_id_uniques, self._time_uniques_all], names=[self._series_id_colname, self._time_colname])
        # Filter only for time indices with a full suite of observations:
        self._time_uniques_full = self._get_times_with_full_observations()
        # TODO: is this efficient, or are we better off indexing using [ , ] instead of levels.isin():
        self._filter_full = self._filter_obs[self._filter_obs.get_level_values(self._time_colname).isin(self._time_uniques_full)]

                                             
    def _get_times_with_full_observations(self):
        count_obs = self._select_df_obs_vs_times().count(level=self._time_colname) 
        full_obs = count_obs[count_obs == self._count_series_indices].dropna()
        time_full = full_obs.index.values
        return time_full
         
        
    # Use the current filter to select a dataframe of all observations
    def _select_df_obs_vs_times(self):
        return self._data_vs_times_df.loc[self._filter_obs, :]
        
        
    # Use the current filter to select a dataframe of a grid of all observations + missing values
    def _select_df_grid_vs_times(self):
        return self._data_vs_times_df.loc[self._filter_grid, :]
        
        
    # Use the current filter to select a dataframe of a grid of only fully-observed times,
    # where fully-observed is defined with respect to the series in this filter (and not globally).
    def _select_df_full_vs_times(self):
        return self._data_vs_times_df.loc[self._filter_full, :]
        
        
    # Convenience method
    def _select_df_vs_times_and_times_switch(self, allow_missing_values):
        if allow_missing_values:
            df_vs_times = self._select_df_grid_vs_times() # return a full grid
            times = self._time_uniques_all
        else:
            df_vs_times = self._select_df_full_vs_times() # return only fully-observed timestamps
            times = self._time_uniques_full
        return (df_vs_times, times)
        
        
    # Convenience method
    def _select_df_vs_series(self):
        if self._data_vs_series_df is None:
            return None
        else:
            df_vs_series = self._data_vs_series_df.loc[self._series_id_uniques]
            return df_vs_series
        
    
    # Returns a 3-item tuple consisting of (a 3-D array of time-label data, a 2-D array of series-label data, a 1D array of time values). 
    # The 3-D array has shape (# series, # timestamps, # time_features) and the 2-D array has shape (# series, # series_features),
    # where #time_features & #series_features depend on the values supplied for value_colnames_filter & include_time_as_feature.
    # Missing observations are replaced by np.nan if applicable.
    def select_arrays(self, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        (df_vs_times, a1d_times) = self._select_df_vs_times_and_times_switch(allow_missing_values=allow_missing_values)
        df_vs_series = self._select_df_vs_series()
        
        # Filter by time-label & series-label names
        selection_count_features = 0
        if value_colnames_filter is None:
            selection_count_features = selection_count_features + self.count_features
        else:
            if type(value_colnames_filter) == str:
                value_colnames_filter = [value_colnames_filter] # for convenience
            self.debug('Filtering by the following value colname(s): ' + str(value_colnames_filter))
            
            # These may be slightly inefficient, but it's important to avoid non-deterministically re-ordering the feature columns, which
            # is what would happpen if we were to do something like this: value_colnames_filter_vs_times = list(set(value_colnames_filter) & set(self._value_colnames_vs_times))
            value_colnames_filter_vs_times = []
            for v in value_colnames_filter:
                if v in self._value_colnames_vs_times:
                    value_colnames_filter_vs_times.append(v)
            
            #set_value_colnames_filter = set(value_colnames_filter)
            #value_colnames_filter_vs_times = list(set_value_colnames_filter & set(self._value_colnames_vs_times))
            df_vs_times = df_vs_times[value_colnames_filter_vs_times]
            selection_count_features = selection_count_features + df_vs_times.shape[1]
            
            if df_vs_series is not None:
                
                # Same NB. as before for this one:
                value_colnames_filter_vs_series = []
                for v in value_colnames_filter:
                    if v in self._value_colnames_vs_series:
                        value_colnames_filter_vs_series.append(v)
                
                df_vs_series = df_vs_series[value_colnames_filter_vs_series]
                selection_count_features = selection_count_features + df_vs_series.shape[1]
        
        # Explicitly include the time column as a feature, if requested
        if include_time_as_feature:
            df_vs_times = df_vs_times.reset_index(level=self._time_colname)
            selection_count_features = selection_count_features + 1
            
        # Prepare the 3-D array of time-label data
        # TODO: is this efficient, or are we better off extracting col-by-col:
        a3d_vs_times = np.array(list(df_vs_times.groupby(level=self._series_id_colname).apply(pd.DataFrame.as_matrix)))
        a3d_vs_times_shape = a3d_vs_times.shape
        self.debug('Converting the filtered dataframe with shape ' + str(df_vs_times.shape) + ' to a 3D array of (series, time, features) with shape ' + str(a3d_vs_times_shape))
        if (len(a3d_vs_times_shape) < 3):
            if len(a3d_vs_times_shape) == 2:
                self.debug('Explicitly Reshaping the 2D time-label array ' + str(a3d_vs_times_shape) + ' to be 3D, since it only has a single feature')
                a3d_vs_times = a3d_vs_times.reshape(a3d_vs_times_shape[0], a3d_vs_times_shape[1], 1)
            else:
                raise Exception('There appear to be no valid time labels, based on the given criteria! The returned array is not 3-dimensional. It has shape ' + str(a3d_vs_times_shape))
           
        # Prepare the 2-D array of series-label data
        if df_vs_series is None:
            a2d_vs_series = None
        else:
            a2d_vs_series = df_vs_series.values
            a2d_vs_series_shape = a2d_vs_series.shape
            if (len(a2d_vs_series_shape) < 2):
                if len(a2d_vs_series_shape) == 1:
                    self.debug('Explicitly Reshaping the 1D series-label array ' + str(a2d_vs_series_shape) + ' to be 2D, since it only has a single feature')
                    a2d_vs_series = a2d_vs_series.reshape(-1, 1) # reshaping to have a single feature, SKLearn-style
                else:
                    raise Exception('There appear to be no valid series labels, based on the given criteria! The returned array is not 2-dimensional. It has shape ' + str(a2d_vs_series_shape))
                
        return (a3d_vs_times, a2d_vs_series, a1d_times)
        
        
    # Returns a 2-item tuple consisting of (a 3-D array of the data, a 1D array of time values). 
    # The 3-D array has shape (# series, # timestamps, # time_features + #series_features)
    # where #time_features & #series_features depend on the values supplied for value_colnames_filter & include_time_as_feature.
    # Missing observations are replaced by np.nan if applicable.
    def select_merged_3d_array(self, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        (a3d_vs_times, a2d_vs_series, a1d_times) = self.select_arrays(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=allow_missing_values)
        #print(a3d_vs_times.shape)   # (186, 31, 1)
        #print(a2d_vs_series.shape)  # (186, 2)
        #print(a1d_times.shape)      # (31,)

        if a2d_vs_series is None:
            a3d_all = a3d_vs_times
        else:
            # Add a time dimension of 1 (in between the series & feature dims)
            a3d_vs_series = a2d_vs_series.reshape((a2d_vs_series.shape[0], 1, a2d_vs_series.shape[1]))
            #print(a3d_vs_series.shape)   # (186, 1, 2)
            
            # Broadcast this new time dimension from a size of 1 to the # timestamps
            a3d_vs_series = np.repeat(a=a3d_vs_series, repeats=a3d_vs_times.shape[1], axis=1)
            #print(a3d_vs_series.shape)   # (186, 31, 2)
            
            a3d_all = np.concatenate([a3d_vs_times, a3d_vs_series], axis=2)
            #print(a3d_all.shape)   # (186, 31, 3)
        
        return (a3d_all, a1d_times)
        
        
    # Returns a 2-item tuple consisting of (a 2-D array of the data, a 1D array of time values). 
    # The 2-D array has shape (# series, # timestamps * # time_features + #series_features)
    # where #time_features & #series_features depend on the values supplied for value_colnames_filter & include_time_as_feature.
    # Missing observations are replaced by np.nan if applicable.
    def select_tabular_full_2d_array(self, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        (a3d_vs_times, a2d_vs_series, a1d_times) = self.select_arrays(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=allow_missing_values)
        #print(a3d_vs_times.shape)   # (186, 31, 1)
        #print(a2d_vs_series.shape)  # (186, 2)
        #print(a1d_times.shape)      # (31,)     
    
        # Flatten over the time & time-feature dimensions
        a2d_vs_times = a3d_vs_times.reshape((a3d_vs_times.shape[0], a3d_vs_times.shape[1] * a3d_vs_times.shape[2]))
        #print(a2d_vs_times.shape)   # (186, 31)
        
        if a2d_vs_series is None:
            a2d_all = a2d_vs_times
        else:
            a2d_all = np.concatenate([a2d_vs_times, a2d_vs_series], axis=1)
            #print(a2d_all.shape)   # (186, 33)
            
        return (a2d_all, a1d_times)
        
    
    # Given an input_window_size and output_window_size, as well as the usual parameters.
    # Returns a 4-item tuple consisting of (a 4-D array of the input X time-labelled data, a 4-D array of the output Y time-labelled data, a 2-D array of series labels, # splits).
    def select_paired_tabular_windowed_arrays(self, input_sliding_window_size, output_sliding_window_size, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # Explicitly convert floats to ints
        input_sliding_window_size = int(input_sliding_window_size)
        output_sliding_window_size = int(output_sliding_window_size)
        
        (a3d_vs_times, a2d_vs_series, a1d_times) = self.select_arrays(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=allow_missing_values)
        
        #print(input_sliding_window_size)    # 3
        #print(output_sliding_window_size)   # 2
        #print(a3d_vs_times.shape)           # (93, 31, 1)
        #print(a2d_vs_series.shape)          # (93, 2)
        
        # Create 2 arrays all_input_time_indices + all_output_time_indices 
        # of shapes (#splits, input_sliding_window_size) + (#splits, output_sliding_window_size), respectively.
        time_splitter = SlidingWindowTimeSeriesSplit(count_timestamps=len(a1d_times), training_set_size=input_sliding_window_size, validation_set_size=output_sliding_window_size, step=1)
        num_splits = len(time_splitter)
        all_input_time_indices = np.empty((num_splits, input_sliding_window_size), dtype=int)
        all_output_time_indices = np.empty((num_splits, output_sliding_window_size), dtype=int)
        split_idx = 0
        for (input_time_indices, output_time_indices) in time_splitter:
            all_input_time_indices[split_idx,:] = input_time_indices
            all_output_time_indices[split_idx,:] = output_time_indices
            split_idx = 1 + split_idx
            
        # Transform the 3-D array of time labels from (# series, # timestamps, # time_features)
        # to pair of 4-D arrays with shapes (# series, # splits, window_size, # time_features)
        # where window size is the input or output window, as appropriate.
        a4d_vs_times_windowed_input = a3d_vs_times[:, all_input_time_indices, :]
        a4d_vs_times_windowed_output = a3d_vs_times[:, all_output_time_indices, :]
        #print(a4d_vs_times_windowed_input.shape)    # (93, 27, 3, 1)
        #print(a4d_vs_times_windowed_output.shape)   # (93, 27, 2, 1)
        
        return (a4d_vs_times_windowed_input, a4d_vs_times_windowed_output, a2d_vs_series, num_splits)
       
    
    # Returns a 2-item tuple of 4-D arrays:
    #   - first item has shape (# series, # splits, input_sliding_window_size, # features)
    #   - second item has shape (# series, # splits, output_sliding_window_size, # features)
    # and # features includes both time-labels and (broadcast) series-labels, where appropriate
    def select_paired_tabular_windowed_4d_arrays(self, input_sliding_window_size, output_sliding_window_size, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        (a4d_vs_times_windowed_input, a4d_vs_times_windowed_output, a2d_vs_series, num_splits) = self.select_paired_tabular_windowed_arrays(input_sliding_window_size=input_sliding_window_size, output_sliding_window_size=output_sliding_window_size, include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=allow_missing_values)
        #print('a4d_vs_times_windowed_input.shape = ' + str(a4d_vs_times_windowed_input.shape))            # (93, 17, 10, 1)
        #print('a4d_vs_times_windowed_output.shape = ' + str(a4d_vs_times_windowed_output.shape))          # (93, 17, 5, 1)
        #print('a2d_vs_series.shape = ' + str(a2d_vs_series.shape))                                        # (93, 2)
        #print('num_splits = ' + str(num_splits))                                                          # 17
        
        if a2d_vs_series is None:
            a4d_vs_times_windowed_input_all = a4d_vs_times_windowed_input
        else:
            # Reshape and then explicitly broadcast along the 2 new axes:
            a4d_vs_series = a2d_vs_series.reshape(a2d_vs_series.shape[0], 1, 1, a2d_vs_series.shape[1])
            a4d_vs_series = np.repeat(a=a4d_vs_series, repeats=num_splits, axis=1)
            a4d_vs_series = np.repeat(a=a4d_vs_series, repeats=input_sliding_window_size, axis=2)
            #print('a4d_vs_series.shape = ' + str(a4d_vs_series.shape))                                        # (93, 17, 10, 2)
            a4d_vs_times_windowed_input_all = np.concatenate([a4d_vs_times_windowed_input, a4d_vs_series], axis=3)
            #print('a4d_vs_times_windowed_input_all.shape = ' + str(a4d_vs_times_windowed_input_all.shape))    # (93, 17, 10, 3)
        
        return (a4d_vs_times_windowed_input_all, a4d_vs_times_windowed_output)
    
    
    # Returns a 2-item tuple of 3-D arrays:
    #   - first item has shape (# series * # splits, input_sliding_window_size, # features)
    #   - second item has shape (# series * # splits, output_sliding_window_size, # features)
    # and # features includes both time-labels and (broadcast) series-labels, where appropriate
    def select_paired_tabular_windowed_3d_by_time_arrays(self, input_sliding_window_size, output_sliding_window_size, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        (a4d_vs_times_windowed_input, a4d_vs_times_windowed_output, a2d_vs_series, num_splits) = self.select_paired_tabular_windowed_arrays(input_sliding_window_size=input_sliding_window_size, output_sliding_window_size=output_sliding_window_size, include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=allow_missing_values)
        a4d_shape_input = a4d_vs_times_windowed_input.shape
        a4d_shape_output = a4d_vs_times_windowed_output.shape
        a3d_vs_times_windowed_input = a4d_vs_times_windowed_input.reshape(a4d_shape_input[0] * a4d_shape_input[1], a4d_shape_input[2], a4d_shape_input[3])
        a3d_vs_times_windowed_output = a4d_vs_times_windowed_output.reshape(a4d_shape_output[0] * a4d_shape_output[1], a4d_shape_output[2], a4d_shape_output[3])
        return (a3d_vs_times_windowed_input, a3d_vs_times_windowed_output)
        
        
    # Given an input_window_size and output_window_size, as well as the usual parameters.
    # Returns a 2-item tuple consisting of (a 3-D array of the input X data, a 3-D array of the output Y data).
    # Each 3-D array will have the shape (#series, #splits, window_size * #time_features + #series_features),
    # where window_size depends on which element of the tuple we are looking at.
    def select_paired_tabular_windowed_3d_by_series_arrays(self, input_sliding_window_size, output_sliding_window_size, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        (a4d_vs_times_windowed_input, a4d_vs_times_windowed_output, a2d_vs_series, num_splits) = self.select_paired_tabular_windowed_arrays(input_sliding_window_size=input_sliding_window_size, output_sliding_window_size=output_sliding_window_size, include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=allow_missing_values)
        
        # And then transform again to 3-D arrays with shapes (# series, # splits, window_size * # time_features)
        a4d_shape_input = a4d_vs_times_windowed_input.shape
        a4d_shape_output = a4d_vs_times_windowed_output.shape
        a3d_vs_times_windowed_input = a4d_vs_times_windowed_input.reshape(a4d_shape_input[0], a4d_shape_input[1], a4d_shape_input[2] * a4d_shape_input[3])
        a3d_vs_times_windowed_output = a4d_vs_times_windowed_output.reshape(a4d_shape_output[0], a4d_shape_output[1], a4d_shape_output[2] * a4d_shape_output[3])
        #print(a3d_vs_times_windowed_input.shape)    # (93, 27, 3)
        #print(a3d_vs_times_windowed_output.shape)   # (93, 27, 2)
        
        if a2d_vs_series is None:
            return (a3d_vs_times_windowed_input, a3d_vs_times_windowed_output)
        else:
            # For the 2-D series, add a split dimension of 1 (in between the series & feature dims)
            a3d_vs_series = a2d_vs_series.reshape((a2d_vs_series.shape[0], 1, a2d_vs_series.shape[1]))
            #print(a3d_vs_series.shape)   # (93, 1, 2)
            
            # Broadcast this new split dimension from a size of 1 to the # splits
            a3d_vs_series = np.repeat(a=a3d_vs_series, repeats=num_splits, axis=1)
            #print(a3d_vs_series.shape)   # (93, 27, 2)
            
            # Finally, combine the series with time data to have arrays of shape:
            # (#series, #splits, window_size * #time_features + #series_features)
            a3d_all_windowed_input = np.concatenate([a3d_vs_times_windowed_input, a3d_vs_series], axis=2)
            a3d_all_windowed_output = np.concatenate([a3d_vs_times_windowed_output, a3d_vs_series], axis=2)
            #print(a3d_all_windowed_input.shape)    # (93, 27, 5)
            #print(a3d_all_windowed_output.shape)   # (93, 27, 4)
            
            return (a3d_all_windowed_input, a3d_all_windowed_output)

        
    # This is a convenience method that converts the results of select_paired_tabular_windowed_2d_arrays(), which are each in shape (#series, #splits, window_size * #time_features + #series_features),
    # to the sklearn-compatible shape of (#series * #splits, window_size * #time_features + #series_features)
    #   i.e. n_samples = #series * #splits
    #   and  n_features = window_size * #time_features + #series_features
    # in sklearn's nomenclature.
    def select_paired_tabular_windowed_2d_arrays(self, input_sliding_window_size, output_sliding_window_size, include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        (a3d_all_windowed_input, a3d_all_windowed_output) = self.select_paired_tabular_windowed_3d_by_series_arrays(input_sliding_window_size=input_sliding_window_size, output_sliding_window_size=output_sliding_window_size, include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=allow_missing_values)
        a3d_shape_input = a3d_all_windowed_input.shape
        a3d_shape_output = a3d_all_windowed_output.shape
        a2d_all_windowed_input  = a3d_all_windowed_input.reshape( (a3d_shape_input[0]  * a3d_shape_input[1],  a3d_shape_input[2] ))
        a2d_all_windowed_output = a3d_all_windowed_output.reshape((a3d_shape_output[0] * a3d_shape_output[1], a3d_shape_output[2]))
        return (a2d_all_windowed_input, a2d_all_windowed_output)
        
        
    # Plot with the time as the X-axis, the series as a group, and the feature as a separate chart.
    def visualise_arrays(self, include_time_as_feature=False, allow_missing_values=False, value_colnames_filter=None):
        (a3d_vs_times, a2d_vs_series, a1d_times) = self.select_arrays(include_time_as_feature=include_time_as_feature, allow_missing_values=allow_missing_values, value_colnames_filter=value_colnames_filter)
        MultiSeries.visualise_external_3d_array(a3d_vs_times, a1d_times)
        MultiSeries.visualise_external_2d_array(a2d_vs_series)
    
        
    # Pythonic (public) static helper method
    @staticmethod
    def visualise_external_3d_array(a3d, times):
        for feature_idx in range(a3d.shape[2]):
            #print(feature_idx)
            fig, ax = plt.subplots()
            for series_idx in range(a3d[:,:,feature_idx].shape[0]):
                arr = a3d[series_idx,:,feature_idx]
                ax.plot(times, arr)
                ax.set_xlabel('time value')
                ax.set_ylabel('value, grouped by series')
                ax.set_title('feature index ' + str(feature_idx))
          
                
    # Pythonic (public) static helper method
    @staticmethod
    def visualise_external_2d_array(a2d):
        if a2d is not None:
            series_idxs = np.arange(a2d.shape[0])
            for feature_idx in range(a2d.shape[1]):
                #print(feature_idx)
                fig, ax = plt.subplots()
                arr = a2d[:,feature_idx]
                ax.scatter(series_idxs, arr)
                ax.set_xlabel('series index')
                ax.set_ylabel('feature value')
                ax.set_title('feature index ' + str(feature_idx))
                          
                
    def visualise(self, title=None, filter_value_colnames=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if filter_value_colnames is None:
            filter_value_colnames = self.all_non_timestamp_feature_names
        elif type(filter_value_colnames) == str:
            filter_value_colnames = [filter_value_colnames] # required
        for value_colname in filter_value_colnames:
            if value_colname in self._value_colnames_vs_times:
                ax = self._select_df_obs_vs_times()[value_colname].unstack().transpose().plot(legend=False, grid=True)
                ax.set_ylabel(value_colname)                
                if title is not None:
                    ax.set_title(title)
            elif value_colname in self._value_colnames_vs_series:
                plt.figure() # to prevent overwriting
                series_df = self._select_df_vs_series()
                plt.grid()
                plt.scatter(x=series_df.index.values, y=series_df[value_colname])
                plt.xlabel('series')
                plt.ylabel(value_colname)  
                if title is not None:
                    plt.title(title)
            else:
                self.warning('Unknown feature ' + value_colname)
                
        
    def visualise_moments(self, title=None, filter_value_colnames=None, stdev_bar_multiple = 1):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if filter_value_colnames is None:
            filter_value_colnames = self._value_colnames_vs_times
        elif type(filter_value_colnames) == str:
            filter_value_colnames = [filter_value_colnames] # required
        axes = []
        for value_colname in filter_value_colnames:
            if value_colname in self._value_colnames_vs_times:
                plt.figure() # force-start a new figure to avoid overlaying
                grouped_series = self._select_df_obs_vs_times()[value_colname].astype('float64').groupby(self._time_colname) # need to explicitly cast because predicted values tend to be of DF type object
                means = grouped_series.mean()
                sd = grouped_series.std(ddof=1) # unbiased (1 degree of freedom)
                n = grouped_series.count()
                #se = sd / np.sqrt(n) 
                #upper_band = means + (stderr_bar_multiple * se)
                #lower_band = means - (stderr_bar_multiple * se)
                upper_band = means + (stdev_bar_multiple * sd)
                lower_band = means - (stdev_bar_multiple * sd)
                ax = upper_band.plot(legend=False, grid=True, color='lightslategray')
                ax = lower_band.plot(legend=False, grid=True, color='lightslategray')
                ax = means.plot(legend=False, grid=True, color='blue') # plot on top of the bands
                #ax = data_weather._select_df_obs_vs_times()[value_colname].unstack().transpose().plot(legend=False, grid=True)
                #ylabel = 'Mean of ' + value_colname + '\n' + '+/- ' + str(stderr_bar_multiple) + ' S.E.'
                ylabel = 'Mean of ' + value_colname + '\n' + '+/- ' + str(stdev_bar_multiple) + ' S.D.'
                ax.set_ylabel(ylabel)                
                if title is not None:
                    ax.set_title(title)
                axes.append(ax)
            elif value_colname in self._value_colnames_vs_series:
                self.warning('Will not plot means for ' + value_colname + ' because it is not a time label')
            else:
                self.warning('Unknown feature ' + value_colname)
        return axes
             
        
    # Returns a tuple (new instance for the given times, new instance for all remaining other times)
    def split_by_times(self, given_times):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        other_times = np.setdiff1d(self._time_uniques_full, given_times)
        if len(other_times) == 0:
            raise Exception('No other times were found beyond the given times: ' + str(given_times))
        new_instance_given_times = self.__new_instance_from_times(new_time_uniques=given_times, require_times_to_be_subset=False)
        new_instance_other_times = self.__new_instance_from_times(new_time_uniques=other_times)
        return (new_instance_given_times, new_instance_other_times)
        
        
    # Returns a single value. Useful for when split_by_times() cannot be called (when we're in the middle of updating the data).
    def subset_by_times(self, given_times):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        new_instance_given_times = self.__new_instance_from_times(new_time_uniques=given_times, require_times_to_be_subset=True)
        return new_instance_given_times
        
        
    def get_backward_time_window(self, window_size, time_limit=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if allow_missing_values:
            timestamps = self._time_uniques_all
        else:
            timestamps = self._time_uniques_full
        count_timestamps = timestamps.shape[0]
        if window_size > count_timestamps:
            raise Exception('There are not enough timestamps for the given parameters window_size=' + str(window_size) + ', allow_missing_values=' + str(allow_missing_values) + '. The ' + str(count_timestamps) + 'available timestamps are ' + str(timestamps))
        
        if time_limit is None:
            subset = timestamps[-window_size:]
        else:
            idx_of_time_limit = np.argwhere(timestamps == time_limit)[0][0]
            subset = timestamps[(idx_of_time_limit - window_size) : idx_of_time_limit]
        
        (new_instance_sliding_window, new_instance_other_times) = self.split_by_times(given_times=subset)
        return new_instance_sliding_window
        
        
    def get_forward_time_window(self, window_size, time_limit=None, allow_missing_values=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if allow_missing_values:
            timestamps = self._time_uniques_all
        else:
            timestamps = self._time_uniques_full
        count_timestamps = timestamps.shape[0]
        if window_size > count_timestamps:
            raise Exception('There are not enough timestamps for the given parameters window_size=' + str(window_size) + ', allow_missing_values=' + str(allow_missing_values) + '. The ' + str(count_timestamps) + 'available timestamps are ' + str(timestamps))
        
        if time_limit is None:
            subset = timestamps[:window_size]
        else:
            idx_of_time_limit = np.argwhere(timestamps == time_limit)[0][0]
            subset = timestamps[(idx_of_time_limit + 1) : (idx_of_time_limit + window_size + 1)]
        
        (new_instance_sliding_window, new_instance_other_times) = self.split_by_times(given_times=subset)
        return new_instance_sliding_window
        
        
    # Utility method to get the internal indices of the boundaries of the given time values - useful for interacting with the statsmodels library.
    def get_time_boundary_indices(self, given_times):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        all_times = self._time_uniques_all
        start_time = np.min(given_times)
        end_time = np.max(given_times)
        start_indices = np.where(all_times == start_time)[0]
        end_indices = np.where(all_times == end_time)[0]
        if len(start_indices) != 1:
            raise Exception('Expected a single instance of time ' + str(start_time) + ' and instead found ' + str(start_indices) + ' in ' + str(all_times))
        if len(end_indices) != 1:
            raise Exception('Expected a single instance of time ' + str(end_time) + ' and instead found ' + str(end_indices) + ' in ' + str(all_times))
        start_index = start_indices[0]
        end_index = end_indices[0]
        return (start_index, end_index)
        
        
    # Generates a tuple of (training MultiSeries object, validation MultiSeries object) using K-fold CV for i.i.d. series
    def generate_series_folds(self, series_splitter=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Yields:
            True if successful, False otherwise.

        """
        if series_splitter is None:
            series_splitter = KFold(n_splits=5)
            self.debug('No seriesSplitter was given, so using ' + str(series_splitter))
        s = 0
        for (training_series_ids_indices, validation_series_ids_indices) in series_splitter.split(self._series_id_uniques):
            s = s + 1
            # Make sure they are sorted so we can use them for MultiIndex slicing later on:
            training_series_ids = np.sort(self._series_id_uniques[training_series_ids_indices])
            validation_series_ids = np.sort(self._series_id_uniques[validation_series_ids_indices])
            count_training_series = len(training_series_ids)
            count_validation_series = len(validation_series_ids)
            self.debug('Split ' + str(s) + ': training/validation series split of ' + str(count_training_series) + '/' + str(count_validation_series) + ' out of ' + str(count_training_series+count_validation_series) + ' total.')
            training_instance = self.__new_instance_from_series_ids(training_series_ids)
            validation_instance = self.__new_instance_from_series_ids(validation_series_ids)
            yield (training_instance, validation_instance)
    
            
    # timeSeriesSplitter should be an implementation of AbstractTimeSeriesSplit
    def generate_time_windows(self, time_splitter):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Yields:
            True if successful, False otherwise.

        """
        for (training_time_indices, validation_time_indices) in time_splitter:
            training_times = self._time_uniques_all[training_time_indices]
            validation_times = self._time_uniques_all[validation_time_indices]
            #count_training_times = len(training_times)
            #count_validation_times = len(validation_times)
            training_instance = self.__new_instance_from_times(training_times)
            validation_instance = self.__new_instance_from_times(validation_times)
            yield (training_instance, validation_instance)
            
            
    # Uses this instance of the MultiSeries & the given times 1D array as a template to convert the given 2D array to a 3D array to a MultiSeries
    def new_instance_from_2d_array(self, Y_a2d, times, prediction_features):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        count_series = Y_a2d.shape[0]
        count_times = len(times)
        count_total_features = Y_a2d.shape[1]
        count_original_features = int(round(count_total_features / count_times)) # enforce integer type due to warning in numpy's reshape() method
        if (count_original_features > 1):
            self.debug('Detected ' + str(count_original_features) + ' original features, will now reshape from a 2D ' + str(Y_a2d.shape) + ' array to a 3D (' + str(count_series) + ', ' + str(count_times) + ', ' + str(count_original_features) + ') array')
        Y_a3d = Y_a2d.reshape(count_series, count_times, count_original_features)
        Y_multiseries = self.new_instance_from_3d_array(a3d_vs_times=Y_a3d, times=times, value_colnames_vs_times=prediction_features)
        return Y_multiseries
        
    
    def _new_data_vs_times_df_from_3d_array(self, times, a3d_vs_times, value_colnames_vs_times, series_id_uniques):
        # Prepare empty data DFs
        new_data_vs_times_df = pd.DataFrame(columns=value_colnames_vs_times, index=pd.MultiIndex.from_product([self._series_id_uniques, times], names=[self._series_id_colname, self._time_colname]))
        #new_data_vs_series_df = pd.DataFrame(columns=value_colnames_vs_series, index=pd.Index(data=self._series_id_uniques, name=self._series_id_colname))
        
        if type(times) == list:
            times_list = times
        else:
            times_list = times.tolist()
        
        # Set the per-time DF with the given values, if specified
        for series_idx in range(a3d_vs_times.shape[0]):
            series_id = series_id_uniques[series_idx]  # look up series ID from the given index
            #self.debug('series_idx=' + str(series_idx) + ', series_id=' + str(series_id))
            # Notes on the next 2 lines, one inside the loop and one outside:
            #   - the only columns in our DF are value cols so this is safe to do
            #   - indexers (inclucing iloc) used for setting do this setting inline
            #   - however, there is no guarantee that new_df itself is a view rather than a copy so we reassign that... refer to
            #       + https://stackoverflow.com/questions/23296282/what-rules-does-pandas-use-to-generate-a-view-vs-a-copy
            #       + http://pandas-docs.github.io/pandas-docs-travis/indexing.html#indexing-view-versus-copy
            new_data_vs_times_df.loc[(series_id, times_list), value_colnames_vs_times] = a3d_vs_times[series_idx,:,:]   
        
        return new_data_vs_times_df
        
    
    def _new_data_vs_times_df_from_2d_array(self, times, series_idx, a2d_vs_times, value_colnames_vs_times, series_id_uniques):
        # Prepare empty data DFs
        new_data_vs_times_df = pd.DataFrame(columns=value_colnames_vs_times, index=pd.MultiIndex.from_product([self._series_id_uniques, times], names=[self._series_id_colname, self._time_colname]))
        #new_data_vs_series_df = pd.DataFrame(columns=value_colnames_vs_series, index=pd.Index(data=self._series_id_uniques, name=self._series_id_colname))
        
        if type(times) == list:
            times_list = times
        else:
            times_list = times.tolist()
        
        # Look up series ID from the given index
        series_id = self._series_id_uniques[series_idx]  
            
        # Set the per-time DF with the given values, if specified
        new_data_vs_times_df.loc[(series_id, times_list), value_colnames_vs_times] = a2d_vs_times  
            
        return new_data_vs_times_df
        
            
    # Instantiate a new instance based off the current instance (Prototype design pattern).
    # Parameter a3d_vs_times should have shape (# series, # timestamps, # features).
    # Parameter "times" should be a subset of the existing prediction times. If it is not, you should use new_mutable_instance() + update_from_3d_array() instead.
    def new_instance_from_3d_array(self, times, a3d_vs_times, value_colnames_vs_times):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
    #def new_instance_from_3d_array(self, times, a3d_vs_times=None, value_colnames_vs_times=None, a2d_vs_series=None, value_colnames_vs_series=None):
        # As well as the copy of the attributes, create an empty copy of the DF
        new_instance = copy.copy(self) # shallow clone
        
        # Infer value_colnames_vs_times
        if value_colnames_vs_times is not None:
            if type(value_colnames_vs_times) == str:
                value_colnames_vs_times = [value_colnames_vs_times] # required
            self.debug('Passed in value_colnames_vs_times = ' + str(value_colnames_vs_times))
        else:
            value_colnames_vs_times = self._data_vs_times_df.columns.values.tolist()
            self.debug('Using default value_colnames_vs_times = ' + str(value_colnames_vs_times))
        new_instance._value_colnames_vs_times = value_colnames_vs_times
        
        # Create a DF from the 3D array...
        new_data_vs_times_df = self._new_data_vs_times_df_from_3d_array(series_id_uniques=self._series_id_uniques, times=times, a3d_vs_times=a3d_vs_times, value_colnames_vs_times=value_colnames_vs_times)
        new_instance._data_vs_times_df = new_data_vs_times_df
          
        # ... and create a new index of observations based off the actual values
        new_instance._filter_obs = new_instance._data_vs_times_df.dropna().index
        self._set_derived_filters()
        
        return new_instance
       
    
    def update_from_3d_array(self, times, a3d_vs_times, value_colnames_vs_times):    
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """    
        # Collect the new data in a similar DF
        new_data_vs_times_df = self._new_data_vs_times_df_from_3d_array(series_id_uniques=self._series_id_uniques, times=times, a3d_vs_times=a3d_vs_times, value_colnames_vs_times=value_colnames_vs_times)
        
        # Set the per-time DF with the given values, if specified
        self._data_vs_times_df.update(new_data_vs_times_df)
                   
        # ... and create a new index of observations based off the actual values
        self._filter_obs = self._data_vs_times_df.dropna().index
        self._set_derived_filters()
            
          
    def update_from_2d_array(self, series_idx, times, a2d_vs_times, value_colnames_vs_times=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # Collect the new data in a similar DF
        new_data_vs_times_df = self._new_data_vs_times_df_from_2d_array(series_idx=series_idx, series_id_uniques=self._series_id_uniques, times=times, a2d_vs_times=a2d_vs_times, value_colnames_vs_times=value_colnames_vs_times)
        
        # Set the per-time DF with the given values, if specified
        self._data_vs_times_df.update(new_data_vs_times_df) 
        
        # ... and create a new index of observations based off the actual values
        self._filter_obs = self._data_vs_times_df.dropna().index
        self._set_derived_filters()
        
    
    # Create a new instance with values at the given prediction_times wiped out. (Other times remain untouched.)
    # This is so we can use this instance to keep track of all the intermediate & final predicitons for various columns.
    def new_mutable_instance(self, filter_value_colnames_vs_times=None, prediction_times=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        new_instance = copy.copy(self) # shallow clone
        
        # Copy whole/subset of time labels DF
        if filter_value_colnames_vs_times is None:
            new_instance._data_vs_times_df = self._select_df_obs_vs_times().copy()
            #new_instance._data_vs_times_df = self._select_df_obs_vs_times()
        else:
            if type(filter_value_colnames_vs_times) == str:
                filter_value_colnames_vs_times = [filter_value_colnames_vs_times]
                new_instance._data_vs_times_df = self._select_df_obs_vs_times()[filter_value_colnames_vs_times].copy()
                new_instance._value_colnames_vs_times = filter_value_colnames_vs_times
                
        # Copy whole series labels DF, if applicable
        series_df = self._select_df_vs_series()
        if series_df is not None:
            new_instance._data_vs_series_df = series_df.copy()
            
        # Add some new times to the given DF index & other internals, so that we can set values on these later on.
        if prediction_times is not None:
            
            additional_times = np.setdiff1d(prediction_times, self._time_uniques_all, assume_unique=True)
            if len(additional_times) > 0:
                self.debug('additional_times = ' + str(additional_times))
                # Append to the times being tracked
                new_instance._time_uniques_all = np.append(arr = self._time_uniques_all, values = additional_times)
                
        new_index = pd.MultiIndex.from_product([self._series_id_uniques, new_instance._time_uniques_all], names=[self._series_id_colname, self._time_colname])
        new_instance._data_vs_times_df = new_instance._data_vs_times_df.reindex(index=new_index)
        new_instance._data_vs_times_df = new_instance._data_vs_times_df.sort_index() # necessary to avoid lexical sort exception when updating later
        
        return new_instance
        
 
    # Call this on Y_true. The residuals are calculated as simply (Y_true - Y_hat) for all dimensions.
    # Returns as an xarray.DataArray.
    # Residuals are added up across all features, so to get the residuals for a single feature, specify it in the value_colnames_filter param.
    # TODO: this only acts on residuals for time labels. Consider whether to expand this to cover series labels too, and if so, how?!
    def get_raw_residuals(self, Y_hat, value_colnames_vs_times_filter=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        Y_true = self # for clarity in the code
        if value_colnames_vs_times_filter is None:
            value_colnames_vs_times_filter = Y_hat._value_colnames_vs_times
        if type(value_colnames_vs_times_filter) == str:
            value_colnames_vs_times_filter = [value_colnames_vs_times_filter] # necessary to avoid implicit conversion of DF -> Series
        residuals_raw_df = (Y_true._select_df_obs_vs_times()[value_colnames_vs_times_filter] - Y_hat._select_df_obs_vs_times()[value_colnames_vs_times_filter])
        # Sum across all features (i.e. value columns) to get overall residuals. This also converts the DF to a Series (still with a MultiIndex):
        residuals_raw_multiseries = residuals_raw_df.sum(axis=1, skipna=True) 
        # Cast this to an xarray.DataArray and set the column names to something standard
        # The next 2 lines aim to make the DataArray as sparse as possible by dropping unused levels on the MultiIndex levels.
        residuals_raw_multiseries.index = residuals_raw_multiseries.index.remove_unused_levels() # requires pandas >= 0.20
        residuals_raw_da = xr.DataArray.from_series(residuals_raw_multiseries) # this sizes the DA as a tensor product of _levels_ not values
        residuals_raw_da = residuals_raw_da.rename({ Y_true._series_id_colname : 'series', Y_true._time_colname : 'timestamp' })
        # Wrap it up in a RawResiduals obj:
        res = RawResiduals(residuals_raw=residuals_raw_da, feature_colnames=value_colnames_vs_times_filter)
        return res        
        
        
    # Instantiate a new instance based off the current instance (Prototype design pattern).
    def __new_instance_from_series_ids(self, new_series_id_uniques):
        # Validation
        missing_ids = np.setdiff1d(new_series_id_uniques, self._series_id_uniques)
        if (missing_ids.size > 0):
            raise Exception(str(missing_ids.size) + ' of the given new_series_id_uniques were not found in the current series IDs, and they are: ' + str(missing_ids))
        # Selection
        # TODO: is this efficient, or are we better off indexing using [ , ] instead of get_level_values().isin():
        new_filter_obs = self._filter_obs[self._filter_obs.get_level_values(self._series_id_colname).isin(new_series_id_uniques)]
        new_time_uniques = new_filter_obs.get_level_values(self._time_colname).unique()
        if type(new_time_uniques) != np.ndarray:
            new_time_uniques = new_time_uniques.values
        new_instance = self.__new_instance_from_filter(new_filter_obs=new_filter_obs, new_series_id_uniques=new_series_id_uniques, new_time_uniques=new_time_uniques)
        return new_instance
        
        
    # Instantiate a new instance based off the current instance (Prototype design pattern).
    def __new_instance_from_times(self, new_time_uniques, require_times_to_be_subset=True):
        # Deal with special case
        if len(new_time_uniques) == 0:
            return None
            
        # Validation, unless "new" times are allowed
        if require_times_to_be_subset:
            missing_ids = np.setdiff1d(new_time_uniques, self._time_uniques_all)
            if (missing_ids.size > 0):
                raise Exception(str(missing_ids.size) + ' of the given new_time_uniques were not found in the current times, and they are: ' + str(missing_ids))
                #self.warning(str(missing_ids.size) + ' of the given new_time_uniques were not found in the current to,es, and they are: ' + str(missing_ids))
                
        # Selection
        # TODO: is this efficient, or are we better off indexing using [ , ] instead of get_level_values().isin():
        new_filter_obs = self._filter_obs[self._filter_obs.get_level_values(self._time_colname).isin(new_time_uniques)]
        new_series_id_uniques = new_filter_obs.get_level_values(self._series_id_colname).unique()
        if type(new_series_id_uniques) != np.ndarray:
            new_series_id_uniques = new_series_id_uniques.values
        new_instance = self.__new_instance_from_filter(new_filter_obs=new_filter_obs, new_series_id_uniques=new_series_id_uniques, new_time_uniques=new_time_uniques)
        return new_instance
    
        
    # Instantiate a new instance based off the current instance (Prototype design pattern).
    def __new_instance_from_filter(self, new_filter_obs, new_series_id_uniques, new_time_uniques):
        new_instance = copy.copy(self) # shallow copy
        new_instance._filter_obs = new_filter_obs
        new_instance._series_id_uniques = new_series_id_uniques
        new_instance._count_series_indices = len(new_series_id_uniques)
        new_instance._time_uniques_all = new_time_uniques
        new_instance._count_time_indices = len(new_time_uniques)
        new_instance._set_derived_filters()
        self.debug('Built new instance ' + str(new_instance))
        return new_instance

        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        rows_filtered = self.count_observations
        rows_total = self._data_vs_times_df.shape[0]
        return (self.__class__.__name__ + '(filtering ' + str(rows_filtered) + '/' + str(rows_total) + ' observations, over ' + str(self._count_series_indices) + ' series x ' + str(self._count_time_indices) + ' timestamps x ' + str(self.count_features) + ' features)')
    

        
  

##################################################
# For testing
##################################################
        
       
if False:
    
    (dummy_vs_times_df, dummy_vs_series_df) = load_dummy_data_df()
    print(dummy_vs_times_df)
    print(dummy_vs_series_df)
    data_dummy = MultiSeries(data_vs_times_df=dummy_vs_times_df, data_vs_series_df=dummy_vs_series_df, time_colname='timestamp', series_id_colnames='series')
    data_dummy.visualise()

if False:
    
    (boston_vs_times_df, boston_vs_series_df) = load_boston_housing_data_df()
    print(boston_vs_times_df)
    print(boston_vs_series_df)
    data_boston = MultiSeries(data_vs_times_df=boston_vs_times_df, data_vs_series_df=boston_vs_series_df, time_colname='feature', series_id_colnames='sample')
    data_boston.visualise()
    
if False:   

    (weather_vs_times_df, weather_vs_series_df) = load_ramsay_weather_data_dfs()
    data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')
        
    (growth_vs_times_df, growth_vs_series_df) = load_ramsay_growth_data_dfs()
    growth_vs_series_df['gender'] = growth_vs_series_df['gender'].astype('category')
    growth_vs_series_df = pd.concat([growth_vs_series_df, pd.get_dummies(growth_vs_series_df['gender'])], axis=1)
    data_growth = MultiSeries(data_vs_times_df=growth_vs_times_df, data_vs_series_df=growth_vs_series_df, time_colname='age', series_id_colnames=['gender', 'cohort_id'])
    
    #data_growth.get_backward_time_window(5, 18).visualise(filter_value_colnames='height')
    
    data_growth.get_backward_time_window(5, 12)._time_uniques_all
    
    data_growth.get_forward_time_window(5, 15.5)._time_uniques_all
    
    
    
    
    data_weather_v2 = data_weather.get_backward_time_window(5).new_mutable_instance()
    data_weather_v2._data_vs_times_df
    
    data_weather_v2._data_vs_times_df.loc[(33,[366]), ['precav', 'tempav']] = [1,2]
    data_weather_v2._data_vs_times_df
    
    data_weather_v2.set_time_labelled_values(prediction_series=[34], prediction_features=['precav', 'tempav'], prediction_times=[366], values=[1,2])
    data_weather_v2._data_vs_times_df
    
    # new_data_vs_times_df.loc[(series_id, times_list), value_colnames_vs_times]
    #data_weather_v2._data_vs_times_df.loc[(33,366),:] = [1,2]
    #data_weather_v2._data_vs_times_df.sort_index()
    
    data_growth_v2 = data_growth.get_backward_time_window(5).new_mutable_instance()
    
    
    
    
    if False: 
        
        data_weather.visualise(title='Weather data')
        data_weather.visualise_means(title='Weather data')
        data_weather.visualise_arrays(include_time_as_feature=True)
        
        data_growth.visualise(title='Growth data')
        data_growth.visualise_means(title='Growth data')
        data_growth.visualise_arrays(include_time_as_feature=True)
        
        
        # Ready for cross-validation
        for (ot, ov) in data_weather.generate_series_folds(series_splitter = KFold(n_splits=5)):
            print('Outer Loop. Training = ' + str(ot._series_id_uniques) + ' / Validation = ' + str(ov._series_id_uniques))
            for (it, iv) in ot.generate_series_folds(series_splitter = KFold(n_splits=5)):
                print('Inner Loop. Training = ' + str(it._series_id_uniques) + ' / Validation = ' + str(iv._series_id_uniques))
                for (st, sv) in it.generate_time_windows(time_splitter = SlidingWindowTimeSeriesSplit(count_timestamps=len(it._time_uniques_all), training_set_size=100, validation_set_size=50, step=50)):
                    print('Timeseries Loop. Training = ' + str(st._time_uniques_all) + ' / Validation = ' + str(sv._time_uniques_all))
                 
                    
        # Ready for cross-validation
        for (ot, ov) in data_growth.generate_series_folds(series_splitter = KFold(n_splits=5)):
            print('Outer Loop. Training = ' + str(ot._series_id_uniques) + ' / Validation = ' + str(ov._series_id_uniques))
            for (it, iv) in ot.generate_series_folds(series_splitter = KFold(n_splits=5)):
                print('Inner Loop. Training = ' + str(it._series_id_uniques) + ' / Validation = ' + str(iv._series_id_uniques))
                for (st, sv) in it.generate_time_windows(time_splitter = SlidingWindowTimeSeriesSplit(count_timestamps=len(it._time_uniques_all), training_set_size=2, validation_set_size=5, step=5)):
                    print('Timeseries Loop. Training = ' + str(st._time_uniques_all) + ' / Validation = ' + str(sv._time_uniques_all))
                    
             
   
       
if False:
    
    # Data: weather
    (weather_vs_times_df, weather_vs_series_df) = load_ramsay_weather_data_dfs()
    data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')
    #data_weather.visualise()
    
    # Data: growth
    (growth_vs_times_df, growth_vs_series_df) = load_ramsay_growth_data_dfs()
    growth_vs_series_df['gender'] = growth_vs_series_df['gender'].astype('category')
    growth_vs_series_df = pd.concat([growth_vs_series_df, pd.get_dummies(growth_vs_series_df['gender'])], axis=1)
    data_growth = MultiSeries(data_vs_times_df=growth_vs_times_df, data_vs_series_df=growth_vs_series_df, time_colname='age', series_id_colnames=['gender', 'cohort_id'])
    #data_growth.visualise()

    input_sliding_window_size = 10
    output_sliding_window_size = 5
    
    (a4d_vs_times_windowed_input, a4d_vs_times_windowed_output) = data_growth.select_paired_tabular_windowed_4d_arrays(input_sliding_window_size=input_sliding_window_size, output_sliding_window_size=output_sliding_window_size)
    print('a4d_vs_times_windowed_input.shape = ' + str(a4d_vs_times_windowed_input.shape))
    print('a4d_vs_times_windowed_output.shape = ' + str(a4d_vs_times_windowed_output.shape))
    
    (a4d_vs_times_windowed_input, a4d_vs_times_windowed_output) = data_weather.select_paired_tabular_windowed_4d_arrays(input_sliding_window_size=input_sliding_window_size, output_sliding_window_size=output_sliding_window_size)
    print('a4d_vs_times_windowed_input.shape = ' + str(a4d_vs_times_windowed_input.shape))
    print('a4d_vs_times_windowed_output.shape = ' + str(a4d_vs_times_windowed_output.shape))
    

if False:
    # Data: weather
    (weather_vs_times_df, weather_vs_series_df) = load_ramsay_weather_data_dfs()
    data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')

    data_weather.visualise()
    
    weather_vs_times_df[weather_vs_times_df.weather_station == 28].set_index('day_of_year')['tempav'].plot()
    weather_vs_times_df[weather_vs_times_df.weather_station == 28].set_index('day_of_year')['precav'].plot()
    
    
              
if False:
    # Data: ECG
    (ecg_vs_times_df, ecg_vs_series_df) = load_ecg_data_dfs()
    data_ecg = MultiSeries(data_vs_times_df=ecg_vs_times_df, data_vs_series_df=ecg_vs_series_df, time_colname='timestamp', series_id_colnames='heartbeat')

    # 133 are normal, the remaining 67 are abnormal.
    ecg_vs_series_df.groupby('is_abnormal').count()
    ecg_vs_times_df['timestamp'].max() # divide into first half (0...47) and second half (48...95)
    
    data_ecg.visualise(filter_value_colnames='potential_difference')
    data_ecg.visualise_moments(filter_value_colnames='potential_difference')
    
    
    
              
if False:
    # Data: Starlight
    (starlight_vs_times_df, starlight_vs_series_df) = load_starlight_data_dfs()
    data_starlight = MultiSeries(data_vs_times_df=starlight_vs_times_df, data_vs_series_df=starlight_vs_series_df, time_colname='folded_time', series_id_colnames='starlight_curve')

    starlight_vs_times_df['folded_time'].max() 
    
    data_starlight.visualise(filter_value_colnames='magnitude')
    data_starlight.visualise_moments(filter_value_colnames='magnitude')
    
    
    
if False:
    
    # Data: Power: Multiple locations & days
    #(power_vs_times_df, power_vs_series_df) = load_power_data_dfs(power_filename='multiple_locations_multiple_days.csv' )
    (power_vs_times_df, power_vs_series_df) = load_power_data_multiple_locations_multiple_days_dfs()
    data_power_multiple_locations_multiple_days = MultiSeries(data_vs_times_df=power_vs_times_df, data_vs_series_df=power_vs_series_df, time_colname='half_hour', series_id_colnames='series_id')
    data_power_multiple_locations_multiple_days.visualise(title='Multiple locations & days')
    data_power_multiple_locations_multiple_days.visualise_moments(title='Multiple locations & days')
    
    # Data: Power: One day, multiple locations
    #(power_vs_times_df, power_vs_series_df) = load_power_data_dfs(power_filename='one_day_multiple_locations.csv' )
    (power_vs_times_df, power_vs_series_df) = load_power_data_one_day_multiple_locations_dfs()
    data_power_one_day_multiple_locations = MultiSeries(data_vs_times_df=power_vs_times_df, data_vs_series_df=power_vs_series_df, time_colname='half_hour', series_id_colnames='series_id')
    data_power_one_day_multiple_locations.visualise(title='One day, multiple locations')
    data_power_one_day_multiple_locations.visualise_moments(title='One day, multiple locations')
    
    # Data: Power: One location, multiple days
    #(power_vs_times_df, power_vs_series_df) = load_power_data_dfs(power_filename='one_location_multiple_days.csv' )
    (power_vs_times_df, power_vs_series_df) = load_power_data_one_location_multiple_days_dfs()
    data_power_one_location_multiple_days = MultiSeries(data_vs_times_df=power_vs_times_df, data_vs_series_df=power_vs_series_df, time_colname='half_hour', series_id_colnames='series_id')
    data_power_one_location_multiple_days.visualise(title='One location, multiple days')
    data_power_one_location_multiple_days.visualise_moments(title='One location, multiple days')
    
    
    
    
    