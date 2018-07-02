
from .logger import LoggingHandler 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr


class RawResiduals(LoggingHandler):
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
    def __init__(self, residuals_raw, feature_colnames):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        super(RawResiduals, self).__init__()
        # Validation
        if residuals_raw is None or feature_colnames is None:
            raise Exception('Cannot pass in a None as either of these params!')
        if type(residuals_raw) != xr.DataArray:
            raise Exception('Was expecting an xarray.DataArray object! Instead was initialised with a ' + str(type(residuals_raw)))
        if type(feature_colnames) == str:
            feature_colnames = [feature_colnames]
        # Assign public fields
        self.residuals_raw = residuals_raw          # xarray.DataArray
        self.feature_colnames = feature_colnames    # list of strings

    def visualise(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        title = 'Raw residuals aggregated over' + '\n' + ' + '.join(self.feature_colnames)
        ax = pd.pivot_table(self.residuals_raw.to_dataframe(name='raw residuals'), index='timestamp', columns='series').plot(legend=False, grid=True, title=title)
        return ax
        
    # For serialization via Pickle
    def __getstate__(self):
        state_dict = {}
        state_dict['feature_colnames'] = self.feature_colnames
        state_dict['residuals_raw'] = self.residuals_raw
        state_dict['feature_colnames'] = self.feature_colnames
        return state_dict
        
    # For deserialization via Pickle
    def __setstate__(self, state):
        state_dict = state
        self.feature_colnames = state_dict['feature_colnames']
        self.residuals_raw = state_dict['residuals_raw']
        self.feature_colnames = state_dict['feature_colnames']
        self.initLogger()

        
class ErrorCurve(LoggingHandler):
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
    def __init__(self):
        super(LoggingHandler, self).__init__()
        # Do nothing. Constructor overloads are not possible in Python, so we initialise using static factory methods.
        
        
    @staticmethod
    def init_from_raw_residuals(raw_residuals_obj):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if type(raw_residuals_obj) != RawResiduals:
            raise Exception('Was expecting a RawResiduals object! Instead was initialised with a ' + str(type(raw_residuals_obj)))

        # Transform the residuals to come up with Loss values for various error metrics.
        # These are xr.DataArray:s with coords of timestamps & series.
        residuals_raw = raw_residuals_obj.residuals_raw
        residuals_abs = np.abs(residuals_raw)   # these are the Loss values for MAE
        residuals_sqr = residuals_raw ** 2      # these are the Loss values for (R)MSE
        
        # Prepare some counts: N = # series, K = # timestamps.
        n = np.max(residuals_raw.count(dim='series').values)
        k = np.max(residuals_raw.count(dim='timestamp').values)
        
        # Calculate the error metrics from the Loss values, per timestamp. These are "1D" xr.DataArray:s.
        mae_per_timestamp = residuals_abs.mean(dim='series')
        mse_per_timestamp = residuals_sqr.mean(dim='series')
        rmse_per_timestamp = np.sqrt(mse_per_timestamp)
        
        # Calculate unbiased (1) estimates instead of biased (0) ones for all (co)variances
        df = 1 
        
        # Calculate SEs (SDs of the mean errors) per timestamp. These are "1D" xr.DataArray:s.
        se_mae_per_timestamp = residuals_abs.std(dim='series', ddof=df) / np.sqrt(n)
        se_mse_per_timestamp = residuals_sqr.std(dim='series', ddof=df) / np.sqrt(n)
        se_rmse_per_timestamp = se_mse_per_timestamp / (2 * rmse_per_timestamp) # Delta-method approx.
        
        # Calculate the error metrics from the Loss values, overall. These are scalars.
        mae_overall = np.mean(residuals_abs.values)
        mse_overall = np.mean(residuals_sqr.values)
        rmse_overall = np.sqrt(mse_overall)
        
        # Calculate sample covariance matrices from the Loss values along the time axis (hence the 
        # transpose). We need these to deal with dependencies along the time axis.
        # These will be K*K matrices.
        covmat_res_abs = np.cov(residuals_abs.values.transpose(), ddof=df)
        covmat_res_sqr = np.cov(residuals_sqr.values.transpose(), ddof=df)
        
        # Calculate SEs (SDs of the mean errors) overall. These are scalars.
        # TODO: I'm using N in the denominator, but is this specifically for the biased or unbiased case?
        se_mae_overall = np.sqrt(np.mean(covmat_res_abs) / n)
        se_mse_overall = np.sqrt(np.mean(covmat_res_sqr) / n)
        se_rmse_overall = se_mse_overall / (2 * rmse_overall) # Delta-method approx.
        
        res = ErrorCurve()
        res._metrics_per_timestamp_values = xr.Dataset({  'mae': mae_per_timestamp
                                                          , 'mse': mse_per_timestamp
                                                          , 'rmse': rmse_per_timestamp
                                                          })
        res._metrics_per_timestamp_stderrs = xr.Dataset({ 'mae': se_mae_per_timestamp
                                                          , 'mse': se_mse_per_timestamp
                                                          , 'rmse': se_rmse_per_timestamp
                                                          })
        res._metrics_overall_values = xr.Dataset({  'mae': xr.DataArray(mae_overall)
                                                  , 'mse': xr.DataArray(mse_overall)
                                                  , 'rmse': xr.DataArray(rmse_overall)
                                                 })
        res._metrics_overall_stderrs = xr.Dataset({  'mae': xr.DataArray(se_mae_overall)
                                                   , 'mse': xr.DataArray(se_mse_overall)
                                                   , 'rmse': xr.DataArray(se_rmse_overall)
                                                  })
        return res
      
    
    @staticmethod
    def init_from_multiple_error_curves(sequence_error_curves):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        list_metrics_overall_values  = [err._metrics_overall_values  for err in sequence_error_curves]
        list_metrics_overall_stderrs = [err._metrics_overall_stderrs for err in sequence_error_curves]
        list_metrics_per_timestamp_values  = [err._metrics_per_timestamp_values  for err in sequence_error_curves]
        list_metrics_per_timestamp_stderrs = [err._metrics_per_timestamp_stderrs for err in sequence_error_curves]
        
        # Gather the folds data together with a new dimension 'fold' (that has not labels)
        concat_metrics_overall_values  = xr.concat(objs=list_metrics_overall_values,  dim='fold')
        concat_metrics_overall_stderrs = xr.concat(objs=list_metrics_overall_stderrs, dim='fold')
        concat_metrics_per_timestamp_values  = xr.concat(objs=list_metrics_per_timestamp_values,  dim='fold')
        concat_metrics_per_timestamp_stderrs = xr.concat(objs=list_metrics_per_timestamp_stderrs, dim='fold')
        
        # Average over all the folds, making sure to take the mean of VARIANCES rather than SDs:
        mean_metrics_overall_values  = concat_metrics_overall_values.mean(dim='fold')
        mean_metrics_overall_stderrs = xr.ufuncs.sqrt((concat_metrics_overall_stderrs ** 2).mean(dim='fold'))
        mean_metrics_per_timestamp_values  = concat_metrics_per_timestamp_values.mean(dim='fold')
        mean_metrics_per_timestamp_stderrs = xr.ufuncs.sqrt((concat_metrics_per_timestamp_stderrs ** 2).mean(dim='fold'))
        
        res = ErrorCurve()
        res._metrics_overall_values = mean_metrics_overall_values
        res._metrics_overall_stderrs = mean_metrics_overall_stderrs
        res._metrics_per_timestamp_values = mean_metrics_per_timestamp_values
        res._metrics_per_timestamp_stderrs = mean_metrics_per_timestamp_stderrs
        return res

        
    def visualise_per_timestamp(self, title=None, metrics = None, stderr_bar_multiple = 1):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # Initial setup
        if metrics is None:
            metrics = ['mae', 'rmse']
        elif type(metrics) == str:
            metrics = [metrics] # for consistency
        if title is None:
            title = ''
        else:
            title = title + '\n'
        title = title + 'Errors per timestamp, with bars representing +/- ' + str(stderr_bar_multiple) + ' S.E.'
        # Extract values, ready for plotting
        values_df = self._metrics_per_timestamp_values.to_dataframe()[metrics]
        stderr_df = self._metrics_per_timestamp_stderrs.to_dataframe()[metrics] * stderr_bar_multiple
        # Fix the x-limits, since the defaults make the error bars illegible
        x_axis_values = self._metrics_per_timestamp_values.coords['timestamp'].values
        x_axis_min_increment = np.min(np.diff(x_axis_values))
        x_axis_min_value = np.min(x_axis_values)
        x_axis_max_value = np.max(x_axis_values)
        xlim_tupe=(x_axis_min_value - x_axis_min_increment/2, x_axis_max_value + x_axis_min_increment/2)
        # Actual plotting
        ax = values_df.plot(yerr=stderr_df, grid=True, xlim=xlim_tupe, capsize=5, title=title)
        ax.legend(bbox_to_anchor=(0, 0, 1.25, 1), loc='right', title='Error metric') # move the legend out of the way
        return ax
    
        
    # Doesn't return anything
    def visualise_overall(self, title=None, metrics=None, stderr_bar_multiple = 1):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # Initial setup
        if metrics is None:
            metrics = ['mae', 'rmse']
        elif type(metrics) == str:
            metrics = [metrics] # for consistency
        if title is None:
            title = ''
        else:
            title = title + '\n'
        title = title + 'Errors overall, with bars representing +/- ' + str(stderr_bar_multiple) + ' S.E.'
      
        values_arr = self._metrics_overall_values[metrics].to_array().to_dataframe(name='value').values.flatten()
        stderr_arr = self._metrics_overall_stderrs[metrics].to_array().to_dataframe(name='value').values.flatten()
        x_vals = np.arange(len(metrics))
        
        # Needed to flush any prev figures, to avoid overwriting
        plt.figure()
        
        # Workaround to remove lines: https://stackoverflow.com/questions/18498742/how-do-you-make-an-errorbar-plot-in-matplotlib-using-linestyle-none-in-rcparams
        plt.plot(x=x_vals, y=values_arr, marker='_')
        plt.errorbar(x=x_vals, y=values_arr, yerr=stderr_arr, fmt='none', capsize=5)
        
        plt.xticks(x_vals, metrics) # this displays strings as X labels
        plt.xlim(np.min(x_vals)-1, np.max(x_vals)+1)
        plt.title(title)
        plt.grid(axis='y')
       
        
    # Returns a long-style DF with columns ['metric_name', 'metric_variable', 'metric_value']
    def get_overall_metrics_as_dataframe(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        df = pd.merge( self._metrics_overall_values.to_array().to_dataframe('value')
                     , self._metrics_overall_stderrs.to_array().to_dataframe('stderr')
              , left_index=True, right_index=True)   
        df = df.reset_index().rename(columns={'variable':'metric_name', 'value':'metric_value', 'stderr':'metric_stderr'})
        return df
    
        
    # Returns a long-style DF with columns ['metric_name', 'timestamp', 'metric_variable', 'metric_value']
    def get_per_timestamp_metrics_as_dataframe(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        df = pd.merge( self._metrics_per_timestamp_values.to_array().to_dataframe('value')
                     , self._metrics_per_timestamp_stderrs.to_array().to_dataframe('stderr')
              , left_index=True, right_index=True)   
        df = df.reset_index().rename(columns={'variable':'metric_name', 'value':'metric_value', 'stderr':'metric_stderr'})
        return df
        
    # For serialization via Pickle
    def __getstate__(self):
        state_dict = {}
        state_dict['_metrics_overall_values'] = self._metrics_overall_values
        state_dict['_metrics_overall_stderrs'] = self._metrics_overall_stderrs
        state_dict['_metrics_per_timestamp_values'] = self._metrics_per_timestamp_values
        state_dict['_metrics_per_timestamp_stderrs'] = self._metrics_per_timestamp_stderrs
        return state_dict
        
    # For deserialization via Pickle
    def __setstate__(self, state):
        state_dict = state
        self._metrics_overall_values = state_dict['_metrics_overall_values']
        self._metrics_overall_stderrs = state_dict['_metrics_overall_stderrs']
        self._metrics_per_timestamp_values = state_dict['_metrics_per_timestamp_values']
        self._metrics_per_timestamp_stderrs = state_dict['_metrics_per_timestamp_stderrs']
        self.initLogger()


##################################################
# For testing
##################################################


# Old testing code removed.
    
    
    
