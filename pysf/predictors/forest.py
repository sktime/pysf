
from .framework import MultiCurveTabularPredictor
from sklearn.ensemble import RandomForestRegressor

# According to this Kaggle page https://www.kaggle.com/general/4092 we should be tuning the following params:
#   1. the number of candidate features (m)
#   2. the depth of the trees (min samples leaf)
#   3. the number of trees



class MultiCurveRandomForestPredictor(MultiCurveTabularPredictor):
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
        MultiCurveTabularPredictor.__init__(self, classic_estimator=RandomForestRegressor(), allow_missing_values=allow_missing_values)
    
    def _replaceWithInt(self, parameter_dict, param_name):
        if param_name in parameter_dict:
            old_val = parameter_dict[param_name]
            parameter_dict[param_name] = int(old_val)
            
    # Override this method so we can validate the inner estimator's parameters
    def set_parameters(self, parameter_dict): 
        # Validate estimator parameters
        if parameter_dict is not None:
            self._replaceWithInt(parameter_dict=parameter_dict, param_name='n_estimators')
        # Call it on the parent
        MultiCurveTabularPredictor.set_parameters(self, parameter_dict=parameter_dict)
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(_classic_estimator = ' + str(self._classic_estimator) + ', allow_missing_values = ' + str(self._allow_missing_values) + ')')
    


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
    # Single run
    ####################################
    
    
    # Init and set hyperparams
    predictor = MultiCurveRandomForestPredictor()
    predictor.set_parameters({'n_estimators' : 20, 'max_features' : 0.90, 'max_depth' : 5})
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
    individual_result.Y_true.visualise(title='Random Forest: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='Random Forest: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='Random Forest, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)
    
    individual_scoring_feature_name = 'precav'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='Random Forest: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='Random Forest: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='Random Forest, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
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
    from sklearn.linear_model import LinearRegression
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
    predictor_templates = {   'Multi-curve self-tuning Random Forest, tuning for precav' : TuningOverallPredictor(predictor_template=MultiCurveRandomForestPredictor(), scoring_metric='rmse', scoring_feature_name='precav'
                                                   , parameter_iterator=ParameterSampler(n_iter=2, param_distributions={  'n_estimators' : [ 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000 ]
                                                                                                                         , 'max_features' : np.arange(0.1, 1.1, 0.1)
                                                                                                                         , 'max_depth' : [ 1, 3, 5, 8 ] 
                                                                                                                         }))
                            }
    combos_of_input_time_column = [ False ]
    combos_of_input_value_colnames = [ None ]
    combos_of_output_value_colnames = [ ['precav'], ['tempav', 'precav'] ]
    evaluator_weather.add_to_targets(predictor_templates=predictor_templates, combos_of_input_time_column=combos_of_input_time_column, combos_of_input_value_colnames=combos_of_input_value_colnames, combos_of_output_value_colnames=combos_of_output_value_colnames)
       
    
    # Tuning for tempav: define targets
    predictor_templates = {   'Multi-curve self-tuning Random Forest, tuning for tempav' : TuningOverallPredictor(predictor_template=MultiCurveRandomForestPredictor(), scoring_metric='rmse', scoring_feature_name='tempav'
                                                   , parameter_iterator=ParameterSampler(n_iter=2, param_distributions={  'n_estimators' : [ 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000 ]
                                                                                                                         , 'max_features' : np.arange(0.1, 1.1, 0.1)
                                                                                                                         , 'max_depth' : [ 1, 3, 5, 8 ] 
                                                                                                                         }))
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



    
    
    
    
