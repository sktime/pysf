
from .framework import AbstractPredictor
from ..logger import LoggingHandler 
from ..errors import ErrorCurve
from ..utils import numpy_to_native

from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection import KFold

import collections
import numpy as np
import pandas as pd
import copy




# Dicts cannot be used as keys because they are mutable. This is an immutable wrapper around a dict.
# Inspired by https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
class FrozenDict(collections.Mapping):
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
    """Don't forget the docstrings!!"""

    def __init__(self, d):
        self._d = dict(d) # Take a copy just in case the original reference is updated.
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        # It would have been simpler and maybe more obvious to 
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of 
        # n we are going to run into, but sometimes it's hard to resist the 
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            self._hash = 0
            for pair in self.__iter__():
                self._hash ^= hash(pair)
        return self._hash
        
    def __repr__(self):
        return str(self._d)
    

class TuningMetrics(LoggingHandler):
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
    def __init__(self, metrics_overall_df, metrics_per_timestamp_df):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        super(TuningMetrics, self).__init__()
        
        # Validation
        if type(metrics_overall_df) != pd.DataFrame:
            raise Exception('Expected the metrics_overall_df parameter to be a pandas DataFrame! Instead it was ' + str(type(metrics_overall_df)))
        if type(metrics_per_timestamp_df) != pd.DataFrame:
            raise Exception('Expected the metrics_per_timestamp_df parameter to be a pandas DataFrame! Instead it was ' + str(type(metrics_per_timestamp_df)))
        
        self.metrics_overall_df = metrics_overall_df
        self.metrics_per_timestamp_df = metrics_per_timestamp_df
       
        
    def _get_filtered_metrics_overall(self, metric, feature_name):
        return self.metrics_overall_df[ (self.metrics_overall_df['metric_name'] == metric) & (self.metrics_overall_df['feature_name'] == feature_name) ]
        
    
    def _get_filtered_metrics_per_timestamp(self, metric, feature_name):
        return self.metrics_per_timestamp_df[ (self.metrics_per_timestamp_df['metric_name'] == metric) & (self.metrics_per_timestamp_df['feature_name'] == feature_name) ]
        
        
    # Returns a 3-tupe of (metric_value, metric_stderr, param_dict) at the optimal/minimal metric value for the given feature
    def get_optimal_params_overall(self, feature_name, metric='rmse'):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        try:
            filtered_metrics_overall_df = self._get_filtered_metrics_overall(metric=metric, feature_name=feature_name)
            min_metric_value = filtered_metrics_overall_df['metric_value'].min()
            min_row = filtered_metrics_overall_df[filtered_metrics_overall_df['metric_value'] == min_metric_value]
            if min_row.shape[0] > 1:
                self.warning('There are ' + str(min_row.shape[0]) + ' sets of parameters that minimise the ' + metric + ' metric for the ' + feature_name + ' feature, so will pick the first. All optima were: \n' + str(min_row))
                min_row = min_row.head(1)
            min_row = min_row.dropna(axis=1) # remove all NaN values for params
            min_metric_stderr = min_row['metric_stderr'].values[0]
            optimal_param_dict = min_row.drop(labels=['metric_name', 'metric_value', 'metric_stderr', 'feature_name'], axis=1).to_dict(orient='records')[0]
            
            safe_optimal_param_dict = {}
            for paramName in optimal_param_dict.keys():
                paramVal = numpy_to_native(optimal_param_dict[paramName])
                safe_optimal_param_dict[paramName] = paramVal
                                              
        except Exception as ex:
            self.error('Exception raised when called for feature_name = ' + str(feature_name) + ', metric = ' + str(metric))
            raise ex # propagate
        return (min_metric_value, min_metric_stderr, safe_optimal_param_dict)
  
    
    def get_optimal_predictors_and_params_per_timestamp(self, feature_name, metric='rmse'):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        filtered_metrics_per_timestamp_df = self._get_filtered_metrics_per_timestamp(metric=metric, feature_name=feature_name).copy()
        
        # In case this is called on a plain 'ol TuningOverallPredictor:
        if not('predictor' in filtered_metrics_per_timestamp_df.columns):
            filtered_metrics_per_timestamp_df['predictor'] = 'default'
            
        min_metric_values_df = filtered_metrics_per_timestamp_df.groupby('timestamp')['metric_value'].min().groupby('timestamp').first().reset_index()
        min_rows_per_timestamp_df = filtered_metrics_per_timestamp_df.merge(min_metric_values_df)
        
        best_predictors_and_params_df = min_rows_per_timestamp_df.drop(labels=['timestamp', 'feature_name', 'metric_name', 'metric_value', 'metric_stderr'], axis=1).drop_duplicates().reset_index(drop=True)
        best_predictors_and_params_df['predictor_id'] = best_predictors_and_params_df.index.values # starting at 0
        min_rows_per_timestamp_with_predictor_ids_df = min_rows_per_timestamp_df.merge(best_predictors_and_params_df)[['timestamp', 'feature_name', 'metric_name', 'metric_value', 'metric_stderr', 'predictor_id']]
    
        list_optimal_predictors_by_id = [None] * len(best_predictors_and_params_df)
        for index, row in best_predictors_and_params_df.iterrows():
            predictor_id = row['predictor_id']
            predictor = row['predictor']
            param_dict = row.drop(['predictor_id', 'predictor']).dropna(axis=0).to_dict()
            list_optimal_predictors_by_id[predictor_id] = (predictor, param_dict)
    
        dict_timestamp_to_optimal_predictor_ids = min_rows_per_timestamp_with_predictor_ids_df[['timestamp', 'predictor_id']].set_index('timestamp').to_dict()['predictor_id']
            
        return (best_predictors_and_params_df, min_rows_per_timestamp_with_predictor_ids_df, dict_timestamp_to_optimal_predictor_ids, list_optimal_predictors_by_id)
        
        
    def boxplot_errors_by_single_param(self, param_name, feature_name, metric='rmse', title=None, convert_param_values_to_string=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        filtered_metrics_overall_df = self._get_filtered_metrics_overall(metric=metric, feature_name=feature_name)
        melted = pd.melt(filtered_metrics_overall_df, id_vars=[param_name, 'metric_name', 'metric_value', 'metric_stderr', 'feature_name'])
        # Deal with unconventional parameter values, like lists:
        if convert_param_values_to_string:
            melted[param_name] = melted[param_name].astype(str)
        # Do the plotting:
        ax = melted[['metric_value', param_name]].boxplot(by=param_name, rot=90)
        ax.set_ylabel(metric)
        ax.set_xlabel(param_name)
        # Clear both titles
        if title is None:
            ax.set_title('Full distribution of cross-validated ' + metric + ' values' + '\n' + 'at values for parameter ' + param_name)
        else:
            ax.set_title(title)
        fig = ax.get_figure()
        fig.suptitle('')
        return ax
    
    def visualise_minimum_errors(self, feature_name, metric='rmse', filter_param_names=None, stderr_bar_multiple=1, title=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # Melt & cast
        filtered_metrics_overall_df = self._get_filtered_metrics_overall(metric=metric, feature_name=feature_name)
        melted = pd.melt(filtered_metrics_overall_df.drop(labels=['metric_name', 'feature_name'], axis=1), id_vars=['metric_value', 'metric_stderr']).rename(columns={'variable' : 'parameter_name', 'value' : 'parameter_value'})
        # TODO: is this right?
        melted['parameter_value'] = pd.to_numeric(melted['parameter_value']) # convert any booleans to numeric values to make plotting work
        melted['metric_stderr'] = melted['metric_stderr'] * stderr_bar_multiple
        
        # Validation, AFTER fetching the data
        if type(filter_param_names) == str:
                filter_param_names = [filter_param_names]                   
        if filter_param_names is None:
            filter_param_names = melted['parameter_name'].unique()
        
        # Get the minimum 'metric_value' vals and their associated 'metric_stderr' vals for each parameter value
        min_metric_vals_per_param_vals = melted.drop(labels=['metric_stderr'], axis=1).groupby(['parameter_name', 'parameter_value']).min().reset_index() # minimise the 'metric_value' field per parameter name + value combo
        min_metric_vals_per_param_vals = pd.merge(melted, min_metric_vals_per_param_vals) # merge on the 'metric_stderr' field
        
        # Chart
        for param_name in filter_param_names:
            sub_df = min_metric_vals_per_param_vals[min_metric_vals_per_param_vals['parameter_name'] == param_name].sort_values(by='parameter_value') # sort so it plots properly
            
            if title is None:
                subtitle = 'Min. cross-validated ' + metric + ' for each value of ' + param_name
            else:
                subtitle = title
                
            # Fix the x-limits, since the defaults make the error bars illegible
            x_axis_values = sub_df['parameter_value'].values
            x_axis_min_increment = np.min(np.diff(x_axis_values))
            x_axis_min_value = np.min(x_axis_values)
            x_axis_max_value = np.max(x_axis_values)
            xlim_tupe=(x_axis_min_value - x_axis_min_increment/2, x_axis_max_value + x_axis_min_increment/2)
            
            #print(sub_df)
            ax = sub_df.plot(x='parameter_value', y='metric_value', yerr='metric_stderr', legend=False, grid=True, xlim=xlim_tupe, capsize=5, title=subtitle)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Minimum cross-validated' + '\n' + metric + ' +/- ' + str(stderr_bar_multiple) + ' S.E.')

    # For serialization via Pickle
    def __getstate__(self):
        state_dict = {}
        state_dict['metrics_overall_df'] = self.metrics_overall_df
        state_dict['metrics_per_timestamp_df'] = self.metrics_per_timestamp_df
        return state_dict
        
    # For deserialization via Pickle
    def __setstate__(self, state):
        state_dict = state
        self.metrics_overall_df = state_dict['metrics_overall_df']
        self.metrics_per_timestamp_df = state_dict['metrics_per_timestamp_df']
        self.initLogger()


class TuningOverallPredictor(AbstractPredictor):
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
    def __init__(self, predictor_template, parameter_iterator, scoring_feature_name, scoring_metric='rmse', series_splitter=None):
        super(TuningOverallPredictor, self).__init__()
        
        # Validation
        if not(issubclass(type(predictor_template), AbstractPredictor)):
            raise Exception('Expected the predictor_template parameter to be a subclass of AbstractPredictor! Instead it was ' + str(type(predictor_template)))
        if not(issubclass(type(parameter_iterator), ParameterGrid)) and not(issubclass(type(parameter_iterator), ParameterSampler)):
            raise Exception('Expected the parameter_iterator parameter to be a subclass of either ParameterGrid or ParameterSampler! Instead it was ' + str(type(parameter_iterator)))
            
        self._predictor_template = predictor_template
        self._parameter_iterator = parameter_iterator
        self._scoring_feature_name = scoring_feature_name
        self._scoring_metric = scoring_metric
        
        # Default to 5-fold CV
        if series_splitter is None:
            self._series_splitter = KFold(n_splits=5)
        else:
            self._series_splitter = series_splitter
            
        # These will be set later
        self.fitted_predictor = None
        self.tuning_metrics = None
        
    def set_parameters(self, parameter_dict):          
        raise Exception('Cannot call set_parameters on an TuningOverallPredictor! parameter_dict = ' + str(parameter_dict))
            
    # Implementation of the abstract method. 
    # Performs the nested cross-validation. Based on the parameters supplied, it decides whether to cross-validated for series and/or timestamps
    def _fitImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # 1. Do the nested cross-validation.
                
        # Perform the nondeterministic sampling (or deterministic grid iteration) and 
        # collect the results into a list of samples to allow reuse.
        param_list = list(self._parameter_iterator) # perform iteration and sampling 
        param_names = sorted(list(param_list[0])) # since the names should be consistent, take them from the very first element
        
        # Contains a dictionary of tuples pointing to lists:
        # (Target object, feature name string) -> list of [ intermediate scoring results ]
        dict_params_and_feature_name_to_list_of_intermediate_scoring_results = {}
        
        # First pass: for common resamples/folds/splits (to be fair in evaluating models against each other!),
        # perform the actual fit & predict and gather up intermediate ScoringResult objects for each (param_dict, feature) combo
        fold_counter = 1
        for (training_instance, validation_instance) in X.generate_series_folds(series_splitter=self._series_splitter):
            self.info('Outer CV Loop. Fold ' + str(fold_counter) + '. Training = ' + str(training_instance._series_id_uniques) + ' / Validation = ' + str(validation_instance._series_id_uniques))
            
            param_dict_counter = 1
            for param_dict in param_list:  
                # Perform the fit & score for this given fold of data and set ()
                self.info('Within fold ' + str(fold_counter) + ', started evaluating param_dict ' + str(param_dict_counter) + '/' + str(len(param_list)) + ': ' + str(param_dict))
                predictor = self._predictor_template.get_deep_copy() # clone to avoid any problems
                predictor.set_parameters(parameter_dict = param_dict) # actually set the params
                predictor.fit(X=training_instance, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
                scoring_results_for_multiple_features = predictor.score(X=validation_instance, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
                predictor.compact()
                
                # For all individual results, construct error curves and cache them in a dictionary
                for feature_name in scoring_results_for_multiple_features:
                    scoring_result = scoring_results_for_multiple_features[feature_name]
                    #scoring_result.err.visualise_per_timestamp(title=('INTERMEDIATE: fold ' + str(fold_counter)))
                    key = (FrozenDict(d=param_dict), feature_name)
                    self.debug('First pass. key = ' + str(key))
                    if not(key in dict_params_and_feature_name_to_list_of_intermediate_scoring_results):
                        dict_params_and_feature_name_to_list_of_intermediate_scoring_results[key] = [] # initialise the list if not already there
                        dict_params_and_feature_name_to_list_of_intermediate_scoring_results[key].append(scoring_result)
                # END: feature_name in scoring_results_for_multiple_features
                
                self.info('Within fold ' + str(fold_counter) + ', finished evaluating param_dict ' + str(param_dict_counter) + '/' + str(len(param_list)) + ': ' + str(param_dict))
                param_dict_counter = param_dict_counter + 1
            # END: for param_dict in param_list
            fold_counter = fold_counter + 1
        # END: for (training_instance, validation_instance) in X.generate_series_folds(series_splitter=self._series_splitter)
     
        # Second pass: for each (param_dict, feature) combo, compute a cross-validated error curve
        tuning_metrics_overall_list = []
        tuning_metrics_per_timestamp_list = []
        for key in dict_params_and_feature_name_to_list_of_intermediate_scoring_results:
            # Retrieve the intermediate error curves and take their averages over all the CV folds
            self.debug('Second pass. key = ' + str(key))
            intermediate_scoring_results = dict_params_and_feature_name_to_list_of_intermediate_scoring_results[key]
            (param_dict, feature_name) = key
            cv_err = ErrorCurve.init_from_multiple_error_curves(sequence_error_curves=[sr.err for sr in intermediate_scoring_results])
            
            # Convert the results to a pair of DFs and store them for later.
            # Make sure that parameter values are stored as objects, to allow mixing of types and objects like lists!
            metrics_overall_df = cv_err.get_overall_metrics_as_dataframe()
            metrics_per_timestamp_df = cv_err.get_per_timestamp_metrics_as_dataframe()
            metrics_overall_df['feature_name'] = feature_name
            metrics_per_timestamp_df['feature_name'] = feature_name
            for param_name in param_names:
                param_val = param_dict[param_name]
                metrics_overall_df.loc[:, param_name] = pd.Series(metrics_overall_df.shape[0] * [ param_val ])              # this allows param_val to be a list
                metrics_overall_df[param_name] = metrics_overall_df[param_name].astype(object)                              # this is to allow mixtures of scalars and lists
                metrics_per_timestamp_df.loc[:, param_name] = pd.Series(metrics_per_timestamp_df.shape[0] * [ param_val ])  # this allows param_val to be a list
                metrics_per_timestamp_df[param_name] = metrics_per_timestamp_df[param_name].astype(object)                  # this is to allow mixtures of scalars and lists
            tuning_metrics_overall_list.append(metrics_overall_df)
            tuning_metrics_per_timestamp_list.append(metrics_per_timestamp_df)
        # END: for key in dict_target_and_feature_name_to_list_of_intermediate_scoring_results
        
        # Concatenate the dataframes into a single list and wrap it all up in a container
        tuning_metrics_overall_df = pd.concat(tuning_metrics_overall_list, ignore_index=True)
        tuning_metrics_per_timestamp_df = pd.concat(tuning_metrics_per_timestamp_list, ignore_index=True)
        self.tuning_metrics = TuningMetrics(metrics_overall_df=tuning_metrics_overall_df, metrics_per_timestamp_df=tuning_metrics_per_timestamp_df)
        
        # 2. Pick the best parameter set to minimise our given feature & given metric (not specific to any timestamps)
        (min_errormetric_overall_value, min_errormetric_overall_stderr, optimal_param_dict) = self.tuning_metrics.get_optimal_params_overall(feature_name=self._scoring_feature_name, metric=self._scoring_metric)
        self.info('Optimal parameters for ' + self._scoring_feature_name + ' ' + self._scoring_metric + ' = ' + str(min_errormetric_overall_value) + ' +/- ' + str(min_errormetric_overall_stderr) + ' are: ' + str(optimal_param_dict))
        
        # 3. Fit over the entire input(X) and assign to self._fitted_predictor
        self.debug('About to fit a predictor with the optimal parameters to the entire input dataset X=' + str(X))
        predictor = self._predictor_template.get_deep_copy() # clone to avoid any problems
        predictor.set_parameters(parameter_dict = optimal_param_dict) # actually set the params
        predictor.fit(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        self.fitted_predictor = predictor
        self.debug('Done fitting the inner predictor with optimal parameters to the entire input dataset: ' + str(self.fitted_predictor))

    
    # Implementation of the abstract method
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        if self.fitted_predictor is None:
            raise Exception('This TuningPredictor must be fitted (i.e. tuned) before we can use it to predict!')
        else:
            Y_hat = self.fitted_predictor._predictImplementation(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
            return Y_hat

    
    # Override of non-abstract method
    def compact(self):
        self.debug('Compacting')
        if self.fitted_predictor is not None:
            self.fitted_predictor.compact()
    
        
    # Implementation of the abstract method.
    def get_deep_copy(self):
        res = TuningOverallPredictor(predictor_template=self._predictor_template, parameter_iterator=self._parameter_iterator, scoring_feature_name=self._scoring_feature_name, scoring_metric=self._scoring_metric, series_splitter=self._series_splitter)
        if self.fitted_predictor is not None:
            res.fitted_predictor = self.fitted_predictor.get_deep_copy()
        if self.tuning_metrics is not None:
            res.tuning_metrics = self.tuning_metrics # a shallow copy is enough since this is immutable
        return res
       
    
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(predictor_template = ' + str(self._predictor_template) + ', parameter_iterator = ' + str(self._parameter_iterator) + ', scoring_feature_name = ' + self._scoring_feature_name + ', scoring_metric = ' + self._scoring_metric + ', series_splitter = ' + str(self._series_splitter) + ', fitted_predictor = ' + str(self.fitted_predictor) + ')')
    
        
        
class TuningTimestampMultiplexerPredictor(AbstractPredictor):
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
    # Parameter list_predictor_template_parameter_iterator_tupes should be a 3-item tuple of the form (key, predictor_template, parameter_iterator)
    def __init__(self, list_predictor_template_parameter_iterator_tupes, scoring_feature_name, scoring_metric='rmse', series_splitter=None):
        super(TuningTimestampMultiplexerPredictor, self).__init__()
        
        # Validation, part 1
        if type(list_predictor_template_parameter_iterator_tupes) != list:
            list_predictor_template_parameter_iterator_tupes = [list_predictor_template_parameter_iterator_tupes]
        
        list_all_template_keys = []
        self._dict_key_to_predictor_templates = {}
        for (template_key, predictor_template, parameter_iterator) in list_predictor_template_parameter_iterator_tupes:
            list_all_template_keys.append(template_key)
            self._dict_key_to_predictor_templates[template_key] = predictor_template
            
        # Validation, part 2
        if len(list_all_template_keys) != len(self._dict_key_to_predictor_templates):
            raise Exception('There is a duplicate template key specified within the list_predictor_template_parameter_iterator_tupes parameter!')
        
        self._list_predictor_template_parameter_iterator_tupes = list_predictor_template_parameter_iterator_tupes
        self._scoring_feature_name = scoring_feature_name
        self._scoring_metric = scoring_metric
        
        # Default to 5-fold CV
        if series_splitter is None:
            self._series_splitter = KFold(n_splits=5)
        else:
            self._series_splitter = series_splitter
            
        # These will be set later
        self.tuning_metrics = None
        self.list_optimal_predictors_by_id = None
        self.dict_timestamp_to_optimal_predictor_ids = None
        self.list_fitted_optimal_predictors_by_id = None
        
    def set_parameters(self, parameter_dict):          
        raise Exception('Cannot call set_parameters on an TuningTimestampMultiplexerPredictor! parameter_dict = ' + str(parameter_dict))
        
    # Implementation of the abstract method. 
    # Performs the nested cross-validation. Based on the parameters supplied, it decides whether to cross-validated for series and/or timestamps
    def _fitImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # 1. Do the nested cross-validation.
        
        # Perform the nondeterministic sampling (or deterministic grid iteration) and collect the results into a list of samples to allow reuse, for each tuple.
        list_expanded_param_vals = [ ( template_key, predictor_template, list(parameter_iterator) ) for (template_key, predictor_template, parameter_iterator) in self._list_predictor_template_parameter_iterator_tupes ]
        list_expanded_param_vals_with_names = [ ( template_key, predictor_template, param_vals_list, sorted(list(param_vals_list[0])) ) for (template_key, predictor_template, param_vals_list) in list_expanded_param_vals ]
        
        # Contains a dictionary of tuples pointing to lists:
        # (template_key, param_dict, feature_name) -> list of [ intermediate scoring results ]
        dict_template_key_params_and_feature_name_to_list_of_intermediate_scoring_results = {}
        
        # First pass: for common resamples/folds/splits (to be fair in evaluating models against each other!),
        # perform the actual fit & predict and gather up intermediate ScoringResult objects for each (param_dict, feature) combo
        fold_counter = 1
        for (training_instance, validation_instance) in X.generate_series_folds(series_splitter=self._series_splitter):
            self.info('Outer CV Loop. Fold ' + str(fold_counter) + '. Training = ' + str(training_instance._series_id_uniques) + ' / Validation = ' + str(validation_instance._series_id_uniques))
            
            template_counter = 1
            for (template_key, predictor_template, param_vals_list, param_names) in list_expanded_param_vals_with_names:
                self.info('Within fold ' + str(fold_counter) + ', started evaluating template_key ' + str(template_counter) + '/' + str(len(list_expanded_param_vals_with_names)) + ': ' + str(template_key))
                
                param_dict_counter = 1
                for param_dict in param_vals_list:  
                    # Perform the fit & score for this given fold of data and set ()
                    self.info('For fold ' + str(fold_counter) + ' + template_key ' + str(template_key) + ', started evaluating param_dict ' + str(param_dict_counter) + '/' + str(len(param_vals_list)) + ': ' + str(param_dict))
                    predictor = predictor_template.get_deep_copy() # clone to avoid any problems
                    predictor.set_parameters(parameter_dict = param_dict) # actually set the params
                    predictor.fit(X=training_instance, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
                    scoring_results_for_multiple_features = predictor.score(X=validation_instance, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
                    # For all individual results, construct error curves and cache them in a dictionary
                    for feature_name in scoring_results_for_multiple_features:
                        scoring_result = scoring_results_for_multiple_features[feature_name]
                        #scoring_result.err.visualise_per_timestamp(title=('INTERMEDIATE: fold ' + str(fold_counter)))
                        key = (template_key, FrozenDict(d=param_dict), feature_name)
                        self.debug('First pass. key = ' + str(key))
                        if not(key in dict_template_key_params_and_feature_name_to_list_of_intermediate_scoring_results):
                            dict_template_key_params_and_feature_name_to_list_of_intermediate_scoring_results[key] = [] # initialise the list if not already there
                            dict_template_key_params_and_feature_name_to_list_of_intermediate_scoring_results[key].append(scoring_result)
                    # END: feature_name in scoring_results_for_multiple_features
                    self.info('For fold ' + str(fold_counter) + ' + template_key ' + str(template_key) + ', finished evaluating param_dict ' + str(param_dict_counter) + '/' + str(len(param_vals_list)) + ': ' + str(param_dict))
                    param_dict_counter = param_dict_counter + 1
                # END: for param_dict in param_list
                
                self.info('Within fold ' + str(fold_counter) + ', finished evaluating template_key ' + str(template_counter) + '/' + str(len(list_expanded_param_vals_with_names)) + ': ' + str(template_key))
                template_counter = template_counter + 1
            # END: for (template_key, predictor_template, param_vals_list, param_names) in list_expanded_param_vals_with_names:
                
            fold_counter = fold_counter + 1
        # END: for (training_instance, validation_instance) in X.generate_series_folds(series_splitter=self._series_splitter)
     
        # Second pass: for each (template_key, param_dict, feature) combo, compute a cross-validated error curve
        tuning_metrics_overall_list = []
        tuning_metrics_per_timestamp_list = []
        for key in dict_template_key_params_and_feature_name_to_list_of_intermediate_scoring_results:
            # Retrieve the intermediate error curves and take their averages over all the CV folds
            self.debug('Second pass. key = ' + str(key))
            intermediate_scoring_results = dict_template_key_params_and_feature_name_to_list_of_intermediate_scoring_results[key]
            (template_key, param_dict, feature_name) = key
            cv_err = ErrorCurve.init_from_multiple_error_curves(sequence_error_curves=[sr.err for sr in intermediate_scoring_results])
            
            # Convert the results to a pair of DFs...
            # Make sure that parameter values are stored as objects, to allow mixing of types and objects like lists!
            metrics_overall_df = cv_err.get_overall_metrics_as_dataframe()
            metrics_per_timestamp_df = cv_err.get_per_timestamp_metrics_as_dataframe()
            metrics_overall_df['feature_name'] = feature_name
            metrics_overall_df['predictor'] = template_key
            metrics_per_timestamp_df['feature_name'] = feature_name
            metrics_per_timestamp_df['predictor'] = template_key        
            for param_name in param_dict:
                param_val = numpy_to_native(param_dict[param_name])
                metrics_overall_df.loc[:, param_name] = pd.Series(metrics_overall_df.shape[0] * [ param_val ])              # this allows param_val to be a list
                metrics_overall_df[param_name] = metrics_overall_df[param_name].astype(object)                              # this is to allow mixtures of scalars and lists
                metrics_per_timestamp_df.loc[:, param_name] = pd.Series(metrics_per_timestamp_df.shape[0] * [ param_val ])  # this allows param_val to be a list
                metrics_per_timestamp_df[param_name] = metrics_per_timestamp_df[param_name].astype(object)                  # this is to allow mixtures of scalars and lists
            # ... and store them for later, knowing that they will be mismatched in terms of col names
            tuning_metrics_overall_list.append(metrics_overall_df)
            tuning_metrics_per_timestamp_list.append(metrics_per_timestamp_df)
        # END: for key in dict_target_and_feature_name_to_list_of_intermediate_scoring_results
        
        # Concatenate the dataframes into a single list and wrap it all up in a container
        tuning_metrics_overall_df = pd.concat(tuning_metrics_overall_list, ignore_index=True)
        tuning_metrics_per_timestamp_df = pd.concat(tuning_metrics_per_timestamp_list, ignore_index=True)
        self.tuning_metrics = TuningMetrics(metrics_overall_df=tuning_metrics_overall_df, metrics_per_timestamp_df=tuning_metrics_per_timestamp_df)
        
        # 2. Pick the best parameter set to minimise our given feature & given metric (not specific to any timestamps)
        (best_predictors_and_params_df, min_rows_per_timestamp_with_predictor_ids_df, dict_timestamp_to_optimal_predictor_ids, list_optimal_predictors_by_id) = self.tuning_metrics.get_optimal_predictors_and_params_per_timestamp(feature_name=self._scoring_feature_name, metric=self._scoring_metric)
        self.list_optimal_predictors_by_id = list_optimal_predictors_by_id
        self.dict_timestamp_to_optimal_predictor_ids = dict_timestamp_to_optimal_predictor_ids
        self.info('Optimal parameters per timestamp and their associated error metrics are as follows:\n' + str(best_predictors_and_params_df))
           
        # TODO: now that we have self.list_optimal_predictors_by_id - can we go through previous predictors and clear part of their memory by calling predictor.compact() ?
        
        # 3. Fit over the entire input(X) for each of predictor/parameter combinations
        count_optimal_predictors = len(self.list_optimal_predictors_by_id)
        self.list_fitted_optimal_predictors_by_id = [None] * count_optimal_predictors
        for predictor_id in range(count_optimal_predictors):
            self.debug('About to fit predictor_id = ' + str(predictor_id) + ' / ' + str(count_optimal_predictors) + ' with the optimal parameters to the entire input dataset X=' + str(X))
            (template_key, optimal_param_dict)  = list_optimal_predictors_by_id[predictor_id]
            predictor = self._dict_key_to_predictor_templates[template_key].get_deep_copy() # clone to avoid any problems
            predictor.set_parameters(parameter_dict = optimal_param_dict) # actually set the params
            predictor.fit(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
            self.list_fitted_optimal_predictors_by_id[predictor_id] = predictor
            self.debug('Done fitting predictor_id = ' + str(predictor_id) + ' / ' + str(count_optimal_predictors) + ' with the optimal parameters to the entire input dataset X=' + str(X))
           
            
    # Implementation of the abstract method
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        if self.list_fitted_optimal_predictors_by_id is None or len(self.list_fitted_optimal_predictors_by_id) == 0:
            raise Exception('This TuningTimestampMultiplexerPredictor must be fitted (i.e. tuned) before we can use it to predict!')
        else:
            count_fitted_optimal_predictors = len(self.list_fitted_optimal_predictors_by_id)
            list_Y_hats_by_predictor_id = [None] * count_fitted_optimal_predictors
            for i in range(count_fitted_optimal_predictors):
                fitted_predictor = self.list_fitted_optimal_predictors_by_id[i]
                self.debug('About to predict from predictor ' + str(1+i) + '/' + str(count_fitted_optimal_predictors) + ' - ' + str(fitted_predictor) )
                list_Y_hats_by_predictor_id[i] = fitted_predictor._predictImplementation(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
                self.debug('Done predicting from predictor ' + str(1+i) + '/' + str(count_fitted_optimal_predictors) + ' - ' + str(fitted_predictor) )
            list_a3d_vs_times = [Y_hat.select_arrays(include_time_as_feature=False, value_colnames_filter=prediction_features, allow_missing_values=False)[0] for Y_hat in list_Y_hats_by_predictor_id]
      
            # Make sure you remove other times, since the new_mutable_instance(..) method will not do that for you:
            multiplexed_Y_hat = X.subset_by_times(prediction_times).new_mutable_instance(filter_value_colnames_vs_times=prediction_features, prediction_times=prediction_times)
            (multiplexed_a3d_vs_times, a2d_vs_series, a1d_times) = multiplexed_Y_hat.select_arrays(include_time_as_feature=False, value_colnames_filter=prediction_features, allow_missing_values=True)
            
            for time_idx in range(len(prediction_times)):
                time_val = prediction_times[time_idx]
                optimal_predictor_id = self.dict_timestamp_to_optimal_predictor_ids[time_val]
                predictor_a3d_vs_times = list_a3d_vs_times[optimal_predictor_id] # has shape (# series, # timestamps, # time_features)
                optimal_predictions_for_timestamp = predictor_a3d_vs_times[:,time_idx,:]
                multiplexed_a3d_vs_times[:,time_idx,:] = optimal_predictions_for_timestamp
            
            multiplexed_Y_hat.update_from_3d_array(times=prediction_times, a3d_vs_times=multiplexed_a3d_vs_times, value_colnames_vs_times=prediction_features)
            return multiplexed_Y_hat
            

    # Implementation of the abstract method.
    def get_deep_copy(self):
        copy_list_predictor_template_parameter_iterator_tupes = [(template_key, predictor_template.get_deep_copy(), parameter_iterator) for (template_key, predictor_template, parameter_iterator) in self._list_predictor_template_parameter_iterator_tupes]
        res = TuningTimestampMultiplexerPredictor(list_predictor_template_parameter_iterator_tupes=copy_list_predictor_template_parameter_iterator_tupes, scoring_feature_name=self._scoring_feature_name, scoring_metric=self._scoring_metric, series_splitter=self._series_splitter)
        
        if self.tuning_metrics is not None:
            res.tuning_metrics = self.tuning_metrics # a shallow copy is enough since this is immutable
        
        if self.list_optimal_predictors_by_id is not None:
            res.list_optimal_predictors_by_id = copy.copy(self.list_optimal_predictors_by_id)
        
        if self.dict_timestamp_to_optimal_predictor_ids is not None:
            res.dict_timestamp_to_optimal_predictor_ids = copy.copy(self.dict_timestamp_to_optimal_predictor_ids)
        
        if self.list_fitted_optimal_predictors_by_id is not None:
            copy_list_fitted_optimal_predictors_by_id = [ predictor.get_deep_copy() for predictor in self.list_fitted_optimal_predictors_by_id ]
            res.list_fitted_optimal_predictors_by_id = copy_list_fitted_optimal_predictors_by_id
        
        return res
    
        
    # Override of non-abstract method
    def compact(self):
        self.debug('Compacting')
        if self.list_optimal_predictors_by_id is not None:
            for predictor in self.list_optimal_predictors_by_id:
                predictor.compact()

            
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(' + str(len(self._list_predictor_template_parameter_iterator_tupes)) + ' predictor/parameter tuples, scoring_feature_name = ' + self._scoring_feature_name + ', scoring_metric = ' + self._scoring_metric + ', series_splitter = ' + str(self._series_splitter) + ')')



        