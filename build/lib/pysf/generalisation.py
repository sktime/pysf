

from .logger import LoggingHandler 
from .data import MultiSeries
from .predictors.framework import AbstractPredictor
from .predictors.tuning import TuningOverallPredictor, TuningTimestampMultiplexerPredictor
from .errors import ErrorCurve
from .utils import get_friendly_list_string, from_pickle

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt




def combine_evaluators(list_of_input_tupes):
    """This is an example of a module level function.
    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.
    Parameter types -- if given -- should be specified according to
    `PEP 484`_, though `PEP 484`_ conformance isn't required or enforced.
    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.
    The format for a parameter is::
        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.
            Multiple paragraphs are supported in parameter
            descriptions.
    Args:
        param1 (int): The first parameter.
        param2 (Optional[str]): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    Returns:
        bool: True if successful, False otherwise.
        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.
        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.
        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::
            {
                'param1': param1,
                'param2': param2
            }
    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.
    .. _PEP 484:
       https://www.python.org/dev/peps/pep-0484/
    """
    res = GeneralisationPerformanceEvaluator(data=None, prediction_times=None)
    res.dict_target_and_feature_name_to_list_of_intermediate_scoring_results = {}
    res.dict_target_to_list_of_tuning_metrics = {}
    for (evaluator_filepath, friendly_keys) in list_of_input_tupes:
        evaluator = from_pickle(evaluator_filepath)
        if type(friendly_keys) == str:
            friendly_keys = [ friendly_keys ]
        for key in evaluator.dict_target_and_feature_name_to_list_of_intermediate_scoring_results:
            (tgt, scoring_field) = key
            tfk = tgt.get_friendly_key()
            if tfk in friendly_keys:
                #print(tgt)
                val_list = evaluator.dict_target_and_feature_name_to_list_of_intermediate_scoring_results[key]
                res.dict_target_and_feature_name_to_list_of_intermediate_scoring_results[key] = val_list
        for tgt in evaluator.dict_target_to_list_of_tuning_metrics:
            tfk = tgt.get_friendly_key()
            if tfk in friendly_keys:
                #print(tgt)
                val_list = evaluator.dict_target_to_list_of_tuning_metrics[tgt]
                res.dict_target_to_list_of_tuning_metrics[tgt] = val_list
    # END for (evaluator_filepath, friendly_keys) in list_of_input_tupes
    res.calculate_second_pass()
    return res
                
    
    
class Target(LoggingHandler):
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
    def __init__(self, data, predictor_template, input_time_column, input_value_colnames, output_value_colnames, description = None):
        super(Target, self).__init__()
        
        # Validation
        if type(data) != MultiSeries:
            raise Exception('Expected the data parameter to be a MultiSeries object! Instead it was ' + str(type(data)))
        if not(issubclass(type(predictor_template), AbstractPredictor)):
            raise Exception('Expected the predictor_template parameter to be a subclass of AbstractPredictor! Instead it was ' + str(type(predictor_template)))
        if description is None:
            description = str(type(predictor_template)) + ' with covariate/s ' + str(input_value_colnames) + ' and response/s ' + str(output_value_colnames)
            
        # Setting public fields
        self.data = data
        self.input_time_column = input_time_column
        self.input_value_colnames = input_value_colnames
        self.output_value_colnames = output_value_colnames
        self.description = description
        
        # Special case for the predictor public field: clone it from the predictor_template parameter, so that there is a unique fittable instance per target.
        # (If we end up fitting/predicting in parallel, then we'll have to do more cloning than this...)
        self.predictor = predictor_template.get_deep_copy()  # deep clone
        
        self.debug('Initialised ' + str(self))
        
    def get_verbose_fields(self):
        inputs = self.input_value_colnames
        if inputs is None:
            inputs = []
        elif type(inputs) == str:
            inputs = [inputs]
        if self.input_time_column:
            inputs = ['time'] + inputs
        outputs = self.output_value_colnames
        if type(outputs) == str:
            outputs = [outputs]
        prediction_inputs = ', '.join(inputs)
        prediction_outputs = ', '.join(outputs)
        verbose_inputs_to_outputs = (prediction_inputs + ' -> ', '')[len(inputs) == 0] + prediction_outputs
        return (prediction_inputs, prediction_outputs, verbose_inputs_to_outputs)
        
    def get_friendly_key(self):
        return (str(self.description) + '/' + str(self.input_time_column) + '/' + get_friendly_list_string(self.input_value_colnames) + '/' + get_friendly_list_string(self.output_value_colnames))
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(' + self.description + '. Use time as Covariate: ' + str(self.input_time_column) +'. Non-time Covariates: ' + str(self.input_value_colnames) + '. Responses: ' + str(self.output_value_colnames) + '. Predictor: ' + str(self.predictor))
    
    # For serialization via Pickle
    def __getstate__(self):
        state_dict = {}
        state_dict['input_time_column'] = self.input_time_column
        state_dict['input_value_colnames'] = self.input_value_colnames
        state_dict['output_value_colnames'] = self.output_value_colnames
        state_dict['description'] = self.description
        return state_dict
        
    # For deserialization via Pickle
    def __setstate__(self, state):
        state_dict = state
        self.input_time_column = state_dict['input_time_column']
        self.input_value_colnames = state_dict['input_value_colnames']
        self.output_value_colnames = state_dict['output_value_colnames']
        self.description = state_dict['description']
        self.predictor = None
        self.initLogger()
        

class GeneralisationPerformance(LoggingHandler):
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
    def __init__(self, target, feature_name, error_curve):
        super(GeneralisationPerformance, self).__init__()
        # Validation
        if type(target) != Target:
            raise Exception('Expected the target parameter to be a Target object! Instead it was ' + str(type(target)))
        if type(feature_name) != str:
            raise Exception('Expected the feature_name parameter to be a str object! Instead it was ' + str(type(feature_name)))
        if type(error_curve) != ErrorCurve:
            raise Exception('Expected the error_curve parameter to be an ErrorCurve object! Instead it was ' + str(type(error_curve)))
        # Setting
        self.target = target
        self.feature_name = feature_name
        self.error_curve = error_curve
        
        
    def get_overall_metrics_df(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        df = self.error_curve.get_overall_metrics_as_dataframe()
        (prediction_inputs, prediction_outputs, verbose_inputs_to_outputs) = self.target.get_verbose_fields()
        df['description'] = self.target.description
        df['feature_name'] = self.feature_name
        df['prediction_inputs'] = prediction_inputs
        df['prediction_outputs'] = prediction_outputs
        df['verbose_inputs_to_outputs'] = verbose_inputs_to_outputs
        return df
        
        
    def get_per_timestamp_metrics_df(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        df = self.error_curve.get_per_timestamp_metrics_as_dataframe()
        (prediction_inputs, prediction_outputs, verbose_inputs_to_outputs) = self.target.get_verbose_fields()
        df['description'] = self.target.description
        df['feature_name'] = self.feature_name
        df['prediction_inputs'] = prediction_inputs
        df['prediction_outputs'] = prediction_outputs
        df['verbose_inputs_to_outputs'] = verbose_inputs_to_outputs
        return df
        
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(target=' + str(self.target) + ', feature_name=' + self.feature_name + ', error_curve=' + str(self.error_curve) + ')')
    
    # For serialization via Pickle
    def __getstate__(self):
        state_dict = {}
        state_dict['target'] = self.target
        state_dict['feature_name'] = self.feature_name
        state_dict['error_curve'] = self.error_curve
        return state_dict
        
    # For deserialization via Pickle
    def __setstate__(self, state):
        state_dict = state
        self.target = state_dict['target']
        self.feature_name = state_dict['feature_name']
        self.error_curve = state_dict['error_curve']
        self.initLogger()
        
        
class GeneralisationPerformanceEvaluator(LoggingHandler):
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
    # The constructor takes in fields that are common to all targets: the data container + prediction times
    def __init__(self, data, prediction_times):
        super(GeneralisationPerformanceEvaluator, self).__init__()
        self.data = data
        self.prediction_times = prediction_times
        self.targets = []
        
        
    # You can call this method multiple times to build up a collection of targets, before evaluating them
    def add_to_targets(self, predictor_templates, combos_of_input_time_column, combos_of_input_value_colnames, combos_of_output_value_colnames):  
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """                  
        # Validation
        if type(combos_of_input_time_column) == bool:
            combos_of_input_time_column = [combos_of_input_time_column]
        if type(combos_of_input_value_colnames) == str or combos_of_input_value_colnames is None:
            combos_of_input_value_colnames = [combos_of_input_value_colnames]
        if type(combos_of_output_value_colnames) == str or combos_of_output_value_colnames is None:
            combos_of_output_value_colnames = [combos_of_output_value_colnames]
        
        # Iteration         
        for predictor_description, predictor_template in predictor_templates.items():
            for input_time_column in combos_of_input_time_column:
                for input_value_colnames in combos_of_input_value_colnames:
                    for output_value_colnames in combos_of_output_value_colnames:
                        target = Target(data=self.data, predictor_template=predictor_template, input_time_column=input_time_column, input_value_colnames=input_value_colnames, output_value_colnames=output_value_colnames, description=predictor_description)
                        self.targets.append(target)
                    
    
    def evaluate(self, series_splitter=None, chart_intermediate_results=False):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        try:
            if series_splitter is None:
                series_splitter = KFold(n_splits=5)
            self.info('Started full evaluation. series_splitter=' + str(series_splitter))
            
            # Contains a dictionary of tuples pointing to lists for all predictors:
            # (Target object, feature name string) -> list of [ intermediate scoring results ]
            self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results = {}
            
            # Contains a dictionary of tuples pointing to lists for the tuning predictors only:
            # (Target object, feature name string) -> list of [ tuning metrics ]
            self.dict_target_to_list_of_tuning_metrics = {}
            
            # First pass: for common resamples/folds/splits (to be fair in evaluating models against each other!),
            # perform the actual fit & predict and gather up intermediate ScoringResult objects for each (target, feature) combo
            fold_counter = 1
            for (training_instance, validation_instance) in self.data.generate_series_folds(series_splitter=series_splitter):
                self.info('Outer CV Loop. Fold ' + str(fold_counter) + '. Training = ' + str(training_instance._series_id_uniques) + ' / Validation = ' + str(validation_instance._series_id_uniques))
                
                if chart_intermediate_results:
                    validation_instance.visualise('Y_true for fold ' + str(fold_counter))
                
                target_counter = 1
                for target in self.targets:
                    self.info('Within fold ' + str(fold_counter) + ', started evaluating individual target ' + str(target_counter) + '/' + str(len(self.targets)) + ': ' + str(target))
                    predictor = target.predictor
                    
                    # Call fit/score
                    try:
                        predictor.fit(X=training_instance, prediction_times=self.prediction_times, input_time_feature=target.input_time_column, input_non_time_features=target.input_value_colnames, prediction_features=target.output_value_colnames)
                        scoring_results_for_multiple_features = predictor.score(X=validation_instance, prediction_times=self.prediction_times, input_time_feature=target.input_time_column, input_non_time_features=target.input_value_colnames, prediction_features=target.output_value_colnames)
                    except Exception as ex2:
                        self.error('Exception while evaluating target ' + str(target) + ' on fold ' + str(fold_counter)) # log where the exception occurred
                        raise ex2 # propagate
                    
                    for feature_name in target.output_value_colnames:
                        scoring_result = scoring_results_for_multiple_features[feature_name]
                        
                        if chart_intermediate_results:
                            scoring_result.Y_hat.visualise(title=('INTERMEDIATE: Y_hat on fold ' + str(fold_counter) + ' for\n' + target.description + ': ' + target.get_verbose_fields()[2]))
                            scoring_result.err.visualise_per_timestamp(title=('INTERMEDIATE: residuals on fold ' + str(fold_counter) + ' for\n' + target.description + ': ' + target.get_verbose_fields()[2]))
                                            
                        key = (target, feature_name)
                        self.debug('First pass for scoring results. key = ' + str(key))
                        if not(key in self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results):
                            self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results[key] = [] # initialise the list if not already there
                        self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results[key].append(scoring_result)
                        
                    # END: for feature_name in target.output_value_colnames
                            
                    key = target
                    self.debug('First pass for tuning metrics. key = ' + str(key))
                    if issubclass(type(predictor), TuningOverallPredictor) or issubclass(type(predictor), TuningTimestampMultiplexerPredictor):
                        if not(key in self.dict_target_to_list_of_tuning_metrics):
                            self.dict_target_to_list_of_tuning_metrics[key] = [] # initialise the list if not already there
                        self.dict_target_to_list_of_tuning_metrics[key].append(predictor.tuning_metrics)
                                
                    predictor.compact()
                    self.info('Within fold ' + str(fold_counter) + ', finished evaluating individual target ' + str(target_counter) + '/' + str(len(self.targets)) + ': ' + str(target))
                    target_counter = target_counter + 1
                # END: for target in self.targets
                fold_counter = fold_counter + 1
            # END: for (training_instance, validation_instance) in self.data.generate_series_folds(series_splitter=series_splitter)
            
            # Second pass: for each (target, feature) combo, compute a cross-validated error curve
            self.calculate_second_pass()
            return self.generalisation_metrics_overall_df
        except Exception as ex1:
            self.error('Logging exception before propagating: ' + str(ex1)) # log where the exception occurred
            raise ex1 # propagate
        
    def calculate_second_pass(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        generalisation_performances = []
        for key in self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results:
            self.debug('Second pass. key = ' + str(key))
            intermediate_scoring_results = self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results[key]
            (target, feature_name) = key
            cv_err = ErrorCurve.init_from_multiple_error_curves(sequence_error_curves=[sr.err for sr in intermediate_scoring_results])
            gen = GeneralisationPerformance(target=target, feature_name=feature_name, error_curve=cv_err)
            generalisation_performances.append(gen)
        # END: for key in dict_target_and_feature_name_to_list_of_intermediate_scoring_results
        
        # Cache the generalisation_performances within the object, and use that to build and store a DF of all 
        # results (and return the latter, for convenience).
        self.info('Finished full evaluation. We now have a total of ' + str(len(generalisation_performances)) + ' individual results.')
        self.generalisation_performances = generalisation_performances
        self.generalisation_metrics_overall_df = pd.concat([ gen.get_overall_metrics_df() for gen in self.generalisation_performances ], ignore_index=True)
        self.generalisation_metrics_per_timestamp_df = pd.concat([ gen.get_per_timestamp_metrics_df() for gen in self.generalisation_performances ], ignore_index=True)
             
        
    def get_sorted_overall_results(self, feature_name, metric = 'rmse'):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        df = self.generalisation_metrics_overall_df.copy()
        df = df[ (df['metric_name'] == metric) & (df['feature_name'] == feature_name) ]
        df = df.sort_values(by='metric_value', axis=0)
        df['rank'] = 1 + np.arange(df.shape[0])
        df['rank_flipped'] = np.flip(df['rank'].values, axis=0)
        return df
        
        
    def get_best_n_overall_results(self, feature_name, best_n_results, metric = 'rmse'):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        df = self.get_sorted_overall_results(feature_name=feature_name, metric=metric)
        if best_n_results is not None:
            df = df[df['rank'] <= best_n_results]
        return df
        
        
    def chart_overall_performance(self, feature_name, metric = 'rmse', best_n_results = None, stderr_bar_multiple = 1, figsize=None, func_update_description_strings=None, color_non_baseline='C0', color_baseline='C3', baseline_regular_expression='Baseline'):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if best_n_results is None:
            title='Showing all results'
        else:
            title='Showing the best ' + str(best_n_results) + ' results'
        
        title = 'Generalisation performance for' + '\n' + 'feature ' + feature_name + ' / ' + 'metric ' + metric + '\n' + title
        
        df = self.get_best_n_overall_results(feature_name=feature_name, best_n_results=best_n_results, metric=metric)
        
        # Update descriptions if necessary
        if func_update_description_strings is not None:
            df['description'] = df['description'].apply(func_update_description_strings)
            
        x_vals = df['metric_value']
        y_vals = df['rank_flipped']
        yticks_vals = df['description'] + '\n' + '(' + df['verbose_inputs_to_outputs'] + ')'
        xerr_vals = df['metric_stderr'] * stderr_bar_multiple
        
        # Needed to flush any prev figures, to avoid overwriting
        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        
        # Plot baselines and non-baselines separately (on the same axes) so that they may be formatted differently
        # FYI, built-in colours are https://matplotlib.org/users/dflt_style_changes.html
        is_baseline = df['description'].str.match(baseline_regular_expression).tolist()
        is_non_baseline = [ not b for b in is_baseline ]
        plt.errorbar(x=x_vals[is_baseline], y=y_vals[is_baseline], xerr=xerr_vals[is_baseline], fmt='o', capsize=5, color=color_baseline)
        plt.errorbar(x=x_vals[is_non_baseline], y=y_vals[is_non_baseline], xerr=xerr_vals[is_non_baseline], fmt='o', capsize=5, color=color_non_baseline)

        plt.yticks(y_vals, yticks_vals)
        plt.xlabel(metric + ' +/- ' + str(stderr_bar_multiple) + ' S.E')
        plt.title(title)
        plt.grid()
        
        
    def chart_per_timestamp_performance(self, feature_name, metric = 'rmse', best_n_overall_results = None, stderr_bar_multiple = 1, errorevery=1, func_update_description_strings=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        title='Showing all results'
        if best_n_overall_results is None:
            best_n_overall_results = np.Inf
            title='Showing all results'
        else:
            title='Showing the best ' + str(best_n_overall_results) + ' overall results'
        
        title = 'Generalisation performance for' + '\n' + 'feature ' + feature_name + ' / ' + 'metric ' + metric + '\n' + title
        
        df = self.generalisation_metrics_per_timestamp_df.copy()
        df = df[ (df['metric_name'] == metric) & (df['feature_name'] == feature_name) ]
        
        # Filter the top N results.
        overall_df = self.get_best_n_overall_results(feature_name=feature_name, best_n_results=best_n_overall_results, metric=metric)
        overall_df = overall_df[['description', 'prediction_inputs', 'prediction_outputs', 'rank', 'rank_flipped']]
        df = df.merge(overall_df)
        
        # Update descriptions if necessary
        if func_update_description_strings is not None:
            df['description'] = df['description'].apply(func_update_description_strings)
        
        count_rank_digits = math.ceil(math.log10(1 + overall_df.shape[0]))
        df['verbose_description'] = df['rank'].astype(str).apply(lambda s: s.zfill(count_rank_digits)) + '. ' + df['description'] + '\n' + '(' + df['verbose_inputs_to_outputs'] + ')'
        df['stderr_bar'] = df['metric_stderr'] * stderr_bar_multiple
        
        # Extract values, ready for plotting
        pivoted_values = pd.pivot_table(df, values='metric_value', columns='verbose_description', index='timestamp')
        pivoted_bars   = pd.pivot_table(df, values='stderr_bar',   columns='verbose_description', index='timestamp')
        # Fix the x-limits, since the defaults make the error bars illegible
        x_axis_values = np.sort(df['timestamp'].unique()) # sort just in case
        x_axis_min_increment = np.min(np.diff(x_axis_values))
        x_axis_min_value = np.min(x_axis_values)
        x_axis_max_value = np.max(x_axis_values)
        xlim_tupe=(x_axis_min_value - x_axis_min_increment/2, x_axis_max_value + x_axis_min_increment/2)
        # Actual plotting    
        ax = pivoted_values.plot(yerr=pivoted_bars, grid=True, capsize=2, xlim=xlim_tupe, title=title, errorevery=errorevery)
        ax.legend(bbox_to_anchor=(0, 0, 1.7, 1), loc='right', prop={'size':8}) 
        ax.set_ylabel(metric + ' +/- ' + str(stderr_bar_multiple) + ' S.E')
        
    def get_intermediate_scoring_results(self, tgt_friendly_key, scoring_feature_name):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        scoring_tgt_features = list(self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results.keys())
        res = []
        for tgt_feature in scoring_tgt_features:
            (tgt, feature) = tgt_feature
            if tgt_friendly_key == tgt.get_friendly_key() and scoring_feature_name == feature:
                list_of_intermediate_scoring_results = self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results[tgt_feature]
                res.append(list_of_intermediate_scoring_results)
        if len(res) > 1:
            self.warning('More than one matching set of intermediate scoring results found: ' + str(len(res)))
        elif len(res) == 0:
            self.warning('No matching set of intermediate scoring results found')
        return res
        
    def get_intermediate_tuning_metrics(self, tgt_friendly_key):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        tuning_tgts = list(self.dict_target_to_list_of_tuning_metrics.keys())
        res = []
        for tgt in tuning_tgts:
            if tgt_friendly_key == tgt.get_friendly_key():
                list_of_tuning_metrics = self.dict_target_to_list_of_tuning_metrics[tgt]
                res.append(list_of_tuning_metrics)
        if len(res) > 1:
            self.warning('More than one matching set of intermediate tuning metrics found: ' + str(len(res)))
        elif len(res) == 0:
            self.warning('No matching set of intermediate tuning metrics found')
        return res
    
    def get_friendly_tgts_keys(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        set_for_scoring_results = set([ tgt.get_friendly_key() for (tgt, scoring_field) in self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results ])
        set_for_tuning_metrics = set([ tgt.get_friendly_key() for tgt in self.dict_target_to_list_of_tuning_metrics ])
        sorted_list_for_scoring_results = sorted(set_for_scoring_results)
        sorted_list_for_tuning_metrics = sorted(set_for_tuning_metrics)
        return (sorted_list_for_scoring_results, sorted_list_for_tuning_metrics)        
        
    def to_csv(self, parent_dirpath):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        evaluator=self
        
        # Save generalisation metrics (each row in the file is a target + metric combo, summarised over all generalisation folds)
        if not os.path.exists(parent_dirpath):
            os.makedirs(parent_dirpath)
        evaluator.generalisation_metrics_overall_df.to_csv(parent_dirpath + 'generalisation_metrics_overall.csv', sep='\t', index=False)
        evaluator.generalisation_metrics_per_timestamp_df.to_csv(parent_dirpath + 'generalisation_metrics_per_timestamp.csv', sep='\t', index=False)
        
        # Save tuning results, where available
        for tgt in evaluator.dict_target_to_list_of_tuning_metrics.keys():
            #print(tgt)
            tgt_dirname = tgt.get_friendly_key().replace('/','!')
            list_of_tuning_metrics = evaluator.dict_target_to_list_of_tuning_metrics[tgt]
            fold = 1
            for tuning_metrics in list_of_tuning_metrics:
                fold_dirname = str(fold)
                fold_dirpath = parent_dirpath + tgt_dirname + '/' + fold_dirname + '/' + 'tuning_metrics/'
                if not os.path.exists(fold_dirpath):
                    os.makedirs(fold_dirpath)
                #print(fold_dirpath)
                tuning_metrics.metrics_overall_df.to_csv(fold_dirpath + 'metrics_overall.csv', sep='\t', index=False)
                tuning_metrics.metrics_per_timestamp_df.to_csv(fold_dirpath + 'metrics_per_timestamp.csv', sep='\t', index=False)
                optimal_params_overall_df = tuning_metrics.metrics_overall_df.loc[tuning_metrics.metrics_overall_df.groupby(['metric_name', 'feature_name'])['metric_value'].idxmin()]
                optimal_params_overall_df.to_csv(fold_dirpath + 'optimal_params_overall.csv', sep='\t', index=False)
                optimal_params_per_timestamp_df = tuning_metrics.metrics_per_timestamp_df.loc[tuning_metrics.metrics_per_timestamp_df.groupby(['metric_name', 'feature_name', 'timestamp'])['metric_value'].idxmin()]
                optimal_params_per_timestamp_df.to_csv(fold_dirpath + 'optimal_params_per_timestamp.csv', sep='\t', index=False)
                fold = fold + 1    
                
        # Save intermediate scoring results
        for key in evaluator.dict_target_and_feature_name_to_list_of_intermediate_scoring_results.keys():
            (tgt, feature_name) = key
            #print(tgt)
            #print(feature_name)
            tgt_dirname = tgt.get_friendly_key().replace('/','!')
            list_of_scoring_results = evaluator.dict_target_and_feature_name_to_list_of_intermediate_scoring_results[key]
            fold = 1
            for scoring_results in list_of_scoring_results:
                fold_dirname = str(fold)
                fold_dirpath = parent_dirpath + tgt_dirname + '/' + fold_dirname + '/'+ 'scoring_results/' + feature_name + '/' 
                if not os.path.exists(fold_dirpath):
                    os.makedirs(fold_dirpath)
                #print(fold_dirpath)
                raw_residuals_df = scoring_results.residuals.residuals_raw.to_dataframe(name='residual').reset_index()
                raw_residuals_df.to_csv(fold_dirpath + 'raw_residuals.csv', sep='\t', index=False)
                error_metrics_overall_df = scoring_results.err.get_overall_metrics_as_dataframe()
                error_metrics_overall_df.to_csv(fold_dirpath + 'error_metrics_overall.csv', sep='\t', index=False)
                error_metrics_per_timestamp_df = scoring_results.err.get_per_timestamp_metrics_as_dataframe()
                error_metrics_per_timestamp_df.to_csv(fold_dirpath + 'error_metrics_per_timestamp.csv', sep='\t', index=False)
                fold = fold + 1
                
        # Return nothing.
        
    # For serialization via Pickle
    def __getstate__(self):
        state_dict = {}
        state_dict['prediction_times'] = self.prediction_times
        state_dict['dict_target_and_feature_name_to_list_of_intermediate_scoring_results'] = self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results
        state_dict['dict_target_to_list_of_tuning_metrics'] = self.dict_target_to_list_of_tuning_metrics
        state_dict['generalisation_performances'] = self.generalisation_performances
        state_dict['generalisation_metrics_overall_df'] = self.generalisation_metrics_overall_df
        state_dict['generalisation_metrics_per_timestamp_df'] = self.generalisation_metrics_per_timestamp_df
        return state_dict
        
        
    # For deserialization via Pickle
    def __setstate__(self, state):
        state_dict = state
        self.prediction_times = state_dict['prediction_times']
        self.dict_target_and_feature_name_to_list_of_intermediate_scoring_results = state_dict['dict_target_and_feature_name_to_list_of_intermediate_scoring_results']
        self.dict_target_to_list_of_tuning_metrics = state_dict['dict_target_to_list_of_tuning_metrics']
        self.generalisation_performances = state_dict['generalisation_performances']
        self.generalisation_metrics_overall_df = state_dict['generalisation_metrics_overall_df']
        self.generalisation_metrics_per_timestamp_df = state_dict['generalisation_metrics_per_timestamp_df']
        self.initLogger()
    

        