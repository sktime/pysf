
from .framework import PipelinePredictor, MultiCurveTabularPredictor
from ..transformers.smoothing import SmoothingSplineTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSCanonical

import copy


def build_multicurve_pcr_predictor():
    # Build a PCA-OLS pipeline:
    classic_estimator_pipeline = Pipeline(steps=[('scale', StandardScaler()), ('pca', PCA()), ('ols', LinearRegression())])
    # Example of how to set the number of components:
    #   classic_estimator.set_params(pca__n_components=3)
    
    # Use that to build a multi-curve predictor
    predictor = MultiCurveTabularPredictor(classic_estimator=classic_estimator_pipeline, allow_missing_values=False)  
    return predictor
    
    
def build_smoothed_multicurve_pcr_predictor():
    return PipelinePredictor(chain = [ SmoothingSplineTransformer() , build_multicurve_pcr_predictor() ])


def build_multicurve_pls_predictor():
    predictor = MultiCurvePlsPredictor()
    return predictor

    
def build_smoothed_multicurve_pls_predictor():
    return PipelinePredictor(chain = [ SmoothingSplineTransformer() , build_multicurve_pls_predictor() ])


class MultiCurvePlsPredictor(MultiCurveTabularPredictor):
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
    def __init__(self, allow_missing_values=False):
        # Explicitly initialise both constructors:
        super(MultiCurvePlsPredictor, self).__init__(classic_estimator=PLSCanonical(), allow_missing_values=allow_missing_values)
            
    # Override this method, just because the classic PLSCanonical.fit(...) has a slightly different signature
    def _fitClassicEstimator(self, X_arr, Y_arr):
        # Fit the sklearn estimator with X in the shape of [n_samples,n_features] taking in our (# series, # times * # features)
        self.debug('About to call fit on the inner ' + str(self._classic_estimator) + ' with the X & Y arrays having the shapes ' + str(X_arr.shape) + ' & ' + str(Y_arr.shape) + ' respectively.')
        self._classic_estimator.fit(X=X_arr, Y=Y_arr) 
          
    def get_deep_copy(self):
        res = MultiCurvePlsPredictor(allow_missing_values=self._allow_missing_values)
        res._classic_estimator = copy.copy(self._classic_estimator)
        return res
            
    

##################################################
# For testing
##################################################


if False:
    
    import pandas as pd
    import numpy as np
    from pysf.data import MultiSeries, load_ramsay_weather_data_dfs, load_ramsay_growth_data_dfs
    from sklearn.model_selection import KFold

    (weather_vs_times_df, weather_vs_series_df) = load_ramsay_weather_data_dfs()
    data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')
        
    #(growth_vs_times_df, growth_vs_series_df) = load_ramsay_growth_data_dfs()
    #growth_vs_series_df['gender'] = growth_vs_series_df['gender'].astype('category')
    #growth_vs_series_df = pd.concat([growth_vs_series_df, pd.get_dummies(growth_vs_series_df['gender'])], axis=1)
    #data_growth = MultiSeries(data_vs_times_df=growth_vs_times_df, data_vs_series_df=growth_vs_series_df, time_colname='age', series_id_colnames=['gender', 'cohort_id'])
 
    # This is a slightly hacky way to generate a single training/test split, since my validation prevents you passing in k=1
    splits = list(data_weather.generate_series_folds(series_splitter=KFold(n_splits=5)))
    (training_instance, validation_instance) = splits[0]
    
    
    # Common target
    include_timestamps_as_features = False
    times = np.arange(301,366)
    prediction_features = ['tempav','precav']
    
    
    ####################################
    # PCA + OLS on weather data
    ####################################
    
    
    # Init and set hyperparams
    predictor = build_smoothed_multicurve_pcr_predictor()
    predictor.set_parameters({'pca__n_components' : 3, 'spline_degree' : 5, 'smoothing_factor' : 100})
    print(predictor)
    
    # TODO: experiment with setting parameters
    #temp = {'pca__n_components' : 3, 'spline_degree' : 5, 'smoothing_factor' : 100}
    #print(temp)
    #valid_keys = list(predictor._chain[1]._classic_estimator.get_params().keys())
    #print(valid_keys)
    #given_keys = list(temp.keys())
    #print(given_keys)
    #invalid_keys = np.setdiff1d(given_keys, valid_keys)
    ##invalid_keys= [ key for key in list(temp.keys()) not in valid_keys]
    #print(invalid_keys)
    #
    #for key in temp:
    #    if key not in valid_keys:
    #        del temp[key]
    #print(temp)
    
    
    predictor.fit(X=training_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    scoring_results = predictor.score(X=validation_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    
    individual_scoring_feature_name = 'tempav'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='Multi-curve PCR: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='Multi-curve PCR: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='Multi-curve PCR, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)
    
    individual_scoring_feature_name = 'precav'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='Multi-curve PCR: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='Multi-curve PCR: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='Multi-curve PCR, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)

    
    
    ####################################
    # PLS on weather data
    ####################################
    
    
    # Init and set hyperparams
    predictor = build_smoothed_multicurve_pls_predictor()
    predictor.set_parameters({'n_components' : 6, 'spline_degree' : 5, 'smoothing_factor' : 100})
    print(predictor)
    
    predictor.fit(X=training_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    scoring_results = predictor.score(X=validation_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    
    individual_scoring_feature_name = 'tempav'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='Multi-curve PLS: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='Multi-curve PLS: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='Multi-curve PLS, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)
    
    individual_scoring_feature_name = 'precav'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='Multi-curve PLS: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='Multi-curve PLS: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='Multi-curve PLS, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)

    
if False:
    
    ####################################
    # Generalisation error estimation
    ####################################
    
    from pysf.data import MultiSeries, load_ramsay_weather_data_dfs
    from pysf.predictors.baselines import SeriesMeansPredictor, ZeroPredictor, TimestampMeansPredictor, SeriesLinearInterpolator
    from pysf.predictors.tuning import TuningOverallPredictor
    from pysf.generalisation import GeneralisationPerformanceEvaluator
    from sklearn.model_selection import ParameterGrid, ParameterSampler
    import numpy as np
    
    (weather_vs_times_df, weather_vs_series_df) = load_ramsay_weather_data_dfs()
    data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')
    #data_weather.visualise()    
      
    
    evaluator_weather = GeneralisationPerformanceEvaluator(data=data_weather, prediction_times=np.arange(301,366))
    
    # Baselines: add targets
    predictor_templates = {   'Baseline single-curve series means' : SeriesMeansPredictor()
                            , 'Baseline 0 values' : ZeroPredictor()
                            , 'Baseline multi-curve timestamp means' : TimestampMeansPredictor()
                            , 'Baseline single-curve series linear interpolator' : SeriesLinearInterpolator()
                          }
    combos_of_input_time_column = [ True ]
    combos_of_input_value_colnames = [ None ]
    combos_of_output_value_colnames = [ ['tempav', 'precav'] ]
    evaluator_weather.add_to_targets(predictor_templates=predictor_templates, combos_of_input_time_column=combos_of_input_time_column, combos_of_input_value_colnames=combos_of_input_value_colnames, combos_of_output_value_colnames=combos_of_output_value_colnames)

    # Single-curve predictors: define targets
    predictor_templates = {   'Multi-curve OLS' :  MultiCurveTabularPredictor(classic_estimator=LinearRegression(), allow_missing_values=False)
                            }
    combos_of_input_time_column = [ False ]
    combos_of_input_value_colnames = [ None ]
    combos_of_output_value_colnames = [ ['precav'], ['tempav'], ['tempav', 'precav'] ]
    evaluator_weather.add_to_targets(predictor_templates=predictor_templates, combos_of_input_time_column=combos_of_input_time_column, combos_of_input_value_colnames=combos_of_input_value_colnames, combos_of_output_value_colnames=combos_of_output_value_colnames)
    
    
    # Tuning for precav: define targets
    predictor_templates = {   'Multi-curve self-tuning Smoothing PCR, tuning for precav' : TuningOverallPredictor(predictor_template=build_smoothed_multicurve_pcr_predictor(), scoring_metric='rmse', scoring_feature_name='precav'
                                                   , parameter_iterator=ParameterSampler(n_iter=15, param_distributions={  'pca__n_components' : [ 3, 5, 7 ]
                                                                                                                         , 'spline_degree' : [ 3, 5 ]
                                                                                                                         , 'smoothing_factor' : [ 'default', '0', '50', '100', '150', '200' ]
                                                                                                                         }))
                            , 'Multi-curve self-tuning Smoothing PLS, tuning for precav' : TuningOverallPredictor(predictor_template=build_smoothed_multicurve_pls_predictor(), scoring_metric='rmse', scoring_feature_name='precav'
                                                   , parameter_iterator=ParameterSampler(n_iter=15, param_distributions={  'n_components' : [ 3, 5, 7 ]
                                                                                                                         , 'spline_degree' : [ 3, 5 ]
                                                                                                                         , 'smoothing_factor' : [ 'default', '0', '50', '100', '150', '200' ]
                                                                                                                         }))
                            , 'Multi-curve self-tuning PCR, tuning for precav' : TuningOverallPredictor(predictor_template=build_multicurve_pcr_predictor(), scoring_metric='rmse', scoring_feature_name='precav'
                                                   , parameter_iterator=ParameterGrid({  'pca__n_components' : [ 3, 5, 7 ] }))
                            , 'Multi-curve self-tuning PLS, tuning for precav' : TuningOverallPredictor(predictor_template=build_multicurve_pls_predictor(), scoring_metric='rmse', scoring_feature_name='precav'
                                                   , parameter_iterator=ParameterGrid({  'n_components' : [ 3, 5, 7 ] }))
                            }
    combos_of_input_time_column = [ False ]
    combos_of_input_value_colnames = [ None ]
    combos_of_output_value_colnames = [ ['precav'], ['tempav', 'precav'] ]
    evaluator_weather.add_to_targets(predictor_templates=predictor_templates, combos_of_input_time_column=combos_of_input_time_column, combos_of_input_value_colnames=combos_of_input_value_colnames, combos_of_output_value_colnames=combos_of_output_value_colnames)
       
    
    # Tuning for tempav: define targets
    predictor_templates = {   'Multi-curve self-tuning Smoothing PCR, tuning for tempav' : TuningOverallPredictor(predictor_template=build_smoothed_multicurve_pcr_predictor(), scoring_metric='rmse', scoring_feature_name='tempav'
                                                   , parameter_iterator=ParameterSampler(n_iter=15, param_distributions={  'pca__n_components' : [ 3, 5, 7 ]
                                                                                                                         , 'spline_degree' : [ 3, 5 ]
                                                                                                                         , 'smoothing_factor' : [ 'default', '0', '50', '100', '150', '200' ]
                                                                                                                         }))
                            , 'Multi-curve self-tuning Smoothing PLS, tuning for tempav' : TuningOverallPredictor(predictor_template=build_smoothed_multicurve_pls_predictor(), scoring_metric='rmse', scoring_feature_name='tempav'
                                                   , parameter_iterator=ParameterSampler(n_iter=15, param_distributions={  'n_components' : [ 3, 5, 7 ]
                                                                                                                         , 'spline_degree' : [ 3, 5 ]
                                                                                                                         , 'smoothing_factor' : [ 'default', '0', '50', '100', '150', '200' ]
                                                                                                                         }))
                            , 'Multi-curve self-tuning PCR, tuning for tempav' : TuningOverallPredictor(predictor_template=build_multicurve_pcr_predictor(), scoring_metric='rmse', scoring_feature_name='tempav'
                                                   , parameter_iterator=ParameterGrid({  'pca__n_components' : [ 3, 5, 7 ] }))
                            , 'Multi-curve self-tuning PLS, tuning for tempav' : TuningOverallPredictor(predictor_template=build_multicurve_pls_predictor(), scoring_metric='rmse', scoring_feature_name='tempav'
                                                   , parameter_iterator=ParameterGrid({  'n_components' : [ 3, 5, 7 ] }))
                            }
    combos_of_input_time_column = [ False ]
    combos_of_input_value_colnames = [ None ]
    combos_of_output_value_colnames = [ ['tempav'], ['tempav', 'precav'] ]
    evaluator_weather.add_to_targets(predictor_templates=predictor_templates, combos_of_input_time_column=combos_of_input_time_column, combos_of_input_value_colnames=combos_of_input_value_colnames, combos_of_output_value_colnames=combos_of_output_value_colnames)
       
    # Do the number-crunching
    generalisation_metrics_overall_df = evaluator_weather.evaluate(chart_intermediate_results=False)
    
    # Chart overall results
    evaluator_weather.chart_overall_performance(feature_name='tempav', metric='rmse', best_n_results=20, figsize=(5, 10))
    evaluator_weather.chart_overall_performance(feature_name='precav', metric='rmse', best_n_results=20, figsize=(5, 10))
    
    # Chart per-timestamp results
    evaluator_weather.chart_per_timestamp_performance(feature_name='tempav', metric='rmse', best_n_overall_results=10)
    #evaluator_weather.chart_per_timestamp_performance(feature_name='tempav', metric='mae', best_n_overall_results=10)
    evaluator_weather.chart_per_timestamp_performance(feature_name='precav', metric='rmse', best_n_overall_results=10)
    #evaluator_weather.chart_per_timestamp_performance(feature_name='precav', metric='mae', best_n_overall_results=15)



    
