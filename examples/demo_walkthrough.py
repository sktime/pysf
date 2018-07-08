

############################################################################
# Exclude from dissertation
############################################################################

# Set up a path to save images
import platform
import matplotlib.pyplot as plt

# Set the current working directory
github_dirpath = ('/Users/Ahmed/Documents/GitHub/', 'E:\\GitHub\\Academic\\')[platform.system() == 'Windows']
image_output_dirpath = github_dirpath + 'ucl-project' + '/dissertation/figures/walkthrough/'
print('image_output_dirpath = '+ str(image_output_dirpath))

save_images = False

############################################################################
# Download & visualise data
############################################################################

from pysf.data import download_ramsay_weather_data_dfs, MultiSeries

# Data: growth
(weather_vs_times_df, weather_vs_series_df) = download_ramsay_weather_data_dfs()
data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')

data_weather.visualise()
data_weather.visualise_moments()

########################################################################################
# Exclude from dissertation
########################################################################################

if save_images:
    data_weather.visualise(filter_value_colnames='tempav')
    plt.savefig(image_output_dirpath + 'data_weather_tempav' + '.pdf')
    
if save_images:
    data_weather.visualise(filter_value_colnames='precav')
    plt.savefig(image_output_dirpath + 'data_weather_precav' + '.pdf')
            
if save_images:
    data_weather.visualise_moments(filter_value_colnames='tempav')
    plt.savefig(image_output_dirpath + 'data_weather_moments_tempav' + '.pdf')
    
if save_images:
    data_weather.visualise_moments(filter_value_colnames='precav')
    plt.savefig(image_output_dirpath + 'data_weather_moments_precav' + '.pdf')    
    

########################################################################################
# Exclude from dissertation
########################################################################################

import random
random.seed(777)

########################################################################################
# Randomly split the data 70%/30 % and visualise
########################################################################################

from sklearn.model_selection import ShuffleSplit

splits = list(data_weather.generate_series_folds(series_splitter=ShuffleSplit(test_size=0.30, n_splits=1)))
(training_set, validation_set) = splits[0]

training_set.visualise(title='Training set')
validation_set.visualise(title='Validation set')

########################################################################################
# Exclude from dissertation
########################################################################################

if save_images:
    training_set.visualise(filter_value_colnames='tempav', title='Training set')
    plt.savefig(image_output_dirpath + 'data_weather_training_tempav' + '.pdf')
    
if save_images:
    training_set.visualise(filter_value_colnames='precav', title='Training set')
    plt.savefig(image_output_dirpath + 'data_weather_training_precav' + '.pdf')
            
if save_images:
    validation_set.visualise(filter_value_colnames='tempav', title='Validation set')
    plt.savefig(image_output_dirpath + 'data_weather_validation_tempav' + '.pdf')
    
if save_images:
    validation_set.visualise(filter_value_colnames='precav', title='Validation set')
    plt.savefig(image_output_dirpath + 'data_weather_validation_precav' + '.pdf')
            

############################################################################
# Build a composite predictor and manually set its hyperparams
############################################################################


from pysf.predictors.framework import PipelinePredictor, MultiCurveTabularPredictor
from pysf.transformers.smoothing import SmoothingSplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

sklearn_estimator_pipeline = Pipeline(steps=[('scale', StandardScaler()), ('pca', PCA()), ('ols', LinearRegression())])
multiseries_smoothing_pcr_predictor = PipelinePredictor(chain = [ SmoothingSplineTransformer() , MultiCurveTabularPredictor(classic_estimator=sklearn_estimator_pipeline)  ])
multiseries_smoothing_pcr_predictor.set_parameters({ 'pca__n_components' : 3 , 'spline_degree' : 5 , 'smoothing_factor' : 'default' })


########################################################################################
# Predict the last 65 days of the year and assess the results.
# We are fitting on the training set and predicting & scoring on the validation set
########################################################################################

from pysf.errors import ErrorCurve
import numpy as np

common_prediction_times = np.arange(301, 366)
common_input_time_feature = False
common_input_non_time_features = ['tempav', 'precav']
common_prediction_features = ['tempav', 'precav']

multiseries_smoothing_pcr_predictor.fit(X=training_set, prediction_times=common_prediction_times, input_time_feature=common_input_time_feature, input_non_time_features=common_input_non_time_features, prediction_features=common_prediction_features)

Y_hat = multiseries_smoothing_pcr_predictor.predict(X=validation_set, prediction_times=common_prediction_times, input_time_feature=common_input_time_feature, input_non_time_features=common_input_non_time_features, prediction_features=common_prediction_features)
Y_hat.visualise(title='Validation set: Y_hat')

Y_true = validation_set.subset_by_times(common_prediction_times)
Y_true.visualise(title='Validation set: Y_true')

for prediction_feature in common_prediction_features:
    residuals = Y_true.get_raw_residuals(Y_hat=Y_hat, value_colnames_vs_times_filter=prediction_feature)
    residuals.visualise()
    err = ErrorCurve.init_from_raw_residuals(raw_residuals_obj=residuals)
    err.visualise_per_timestamp(title=prediction_feature)


########################################################################################
# Exclude from dissertation
########################################################################################

if save_images:
    Y_hat.visualise(title='Validation set: Y_hat', filter_value_colnames='tempav')
    plt.savefig(image_output_dirpath + 'predicting_fixed_Y_hat_tempav' + '.pdf')
    
if save_images:
    Y_hat.visualise(title='Validation set: Y_hat', filter_value_colnames='precav')
    plt.savefig(image_output_dirpath + 'predicting_fixed_Y_hat_precav' + '.pdf')    

if save_images:
    Y_true.visualise(title='Validation set: Y_true', filter_value_colnames='tempav')
    plt.savefig(image_output_dirpath + 'predicting_fixed_Y_true_tempav' + '.pdf')
    
if save_images:
    Y_true.visualise(title='Validation set: Y_true', filter_value_colnames='precav')
    plt.savefig(image_output_dirpath + 'predicting_fixed_Y_true_precav' + '.pdf')    

temp_feature = 'tempav'
if save_images:
    residuals = Y_true.get_raw_residuals(Y_hat=Y_hat, value_colnames_vs_times_filter=temp_feature)
    residuals.visualise()
    plt.savefig(image_output_dirpath + 'predicting_fixed_residuals_' + temp_feature + '.pdf') 

if save_images:
    err = ErrorCurve.init_from_raw_residuals(raw_residuals_obj=residuals)
    err.visualise_per_timestamp(title=temp_feature)
    plt.legend(loc='upper left', prop={'size':8}) # bbox_to_anchor=(0, 0, 1.86, 1), 
    plt.savefig(image_output_dirpath + 'predicting_fixed_errors_' + temp_feature + '.pdf') 
    
temp_feature = 'precav'
if save_images:
    residuals = Y_true.get_raw_residuals(Y_hat=Y_hat, value_colnames_vs_times_filter=temp_feature)
    residuals.visualise()
    plt.savefig(image_output_dirpath + 'predicting_fixed_residuals_' + temp_feature + '.pdf') 

if save_images:
    err = ErrorCurve.init_from_raw_residuals(raw_residuals_obj=residuals)
    err.visualise_per_timestamp(title=temp_feature)
    plt.legend(loc='upper left', prop={'size':8}) # bbox_to_anchor=(0, 0, 1.86, 1), 
    plt.savefig(image_output_dirpath + 'predicting_fixed_errors_' + temp_feature + '.pdf') 
    
########################################################################################
# Define a tuning predictor (tuning for overall error), wrapped around the given 
# multicurve smoothing PCR predictor
# Our definition means we willsample 6 parameter sets at random, the fitting step will 
# performs 5-fold cross-validation (the default), and will tune for tempav even though
# we may choose to predict for both features concurrently. 
########################################################################################
    
from pysf.predictors.tuning import TuningOverallPredictor
from sklearn.model_selection import ParameterSampler

tuning_overall_multiseries_smoothing_pcr_predictor = TuningOverallPredictor(predictor_template=multiseries_smoothing_pcr_predictor, scoring_metric='rmse', scoring_feature_name='tempav'
                                                   , parameter_iterator=ParameterSampler(n_iter=5, param_distributions={  'pca__n_components' : [ 3, 5, 7, 9 ]
                                                                                                                         , 'spline_degree' : [ 3, 5 ]
                                                                                                                         , 'smoothing_factor' : [ 'default', '0', '50', '100', '150', '200' ]
                                                                                                                         }))

########################################################################################
# Fit and predict. The fit step will estimate optimal hyperparmater settings.
# You can see from the scoring that predictive performance has improved.
########################################################################################

import random
random.seed(777) # for reproducibility

tuning_overall_multiseries_smoothing_pcr_predictor.fit(X=training_set, prediction_times=common_prediction_times, input_time_feature=common_input_time_feature, input_non_time_features=common_input_non_time_features, prediction_features=common_prediction_features)

Y_hat = tuning_overall_multiseries_smoothing_pcr_predictor.predict(X=validation_set, prediction_times=common_prediction_times, input_time_feature=common_input_time_feature, input_non_time_features=common_input_non_time_features, prediction_features=common_prediction_features)
Y_hat.visualise(title='Validation set: Y_hat')

for prediction_feature in common_prediction_features:
    residuals = Y_true.get_raw_residuals(Y_hat=Y_hat, value_colnames_vs_times_filter=prediction_feature)
    residuals.visualise()
    err = ErrorCurve.init_from_raw_residuals(raw_residuals_obj=residuals)
    err.visualise_per_timestamp(title=prediction_feature)

    
########################################################################################
# Exclude from dissertation
########################################################################################


if save_images:
    Y_hat.visualise(title='Validation set: Y_hat', filter_value_colnames='tempav')
    plt.savefig(image_output_dirpath + 'predicting_tuning_Y_hat_tempav' + '.pdf')
    
if save_images:
    Y_hat.visualise(title='Validation set: Y_hat', filter_value_colnames='precav')
    plt.savefig(image_output_dirpath + 'predicting_tuning_Y_hat_precav' + '.pdf')    

if save_images:
    Y_true.visualise(title='Validation set: Y_true', filter_value_colnames='tempav')
    plt.savefig(image_output_dirpath + 'predicting_tuning_Y_true_tempav' + '.pdf')
    
if save_images:
    Y_true.visualise(title='Validation set: Y_true', filter_value_colnames='precav')
    plt.savefig(image_output_dirpath + 'predicting_tuning_Y_true_precav' + '.pdf')    

temp_feature = 'tempav'
if save_images:
    residuals = Y_true.get_raw_residuals(Y_hat=Y_hat, value_colnames_vs_times_filter=temp_feature)
    residuals.visualise()
    plt.savefig(image_output_dirpath + 'predicting_tuning_residuals_' + temp_feature + '.pdf') 

if save_images:
    err = ErrorCurve.init_from_raw_residuals(raw_residuals_obj=residuals)
    err.visualise_per_timestamp(title=temp_feature)
    plt.legend(loc='upper left', prop={'size':8}) # bbox_to_anchor=(0, 0, 1.86, 1), 
    plt.savefig(image_output_dirpath + 'predicting_tuning_errors_' + temp_feature + '.pdf') 
    
temp_feature = 'precav'
if save_images:
    residuals = Y_true.get_raw_residuals(Y_hat=Y_hat, value_colnames_vs_times_filter=temp_feature)
    residuals.visualise()
    plt.savefig(image_output_dirpath + 'predicting_tuning_residuals_' + temp_feature + '.pdf') 

if save_images:
    err = ErrorCurve.init_from_raw_residuals(raw_residuals_obj=residuals)
    err.visualise_per_timestamp(title=temp_feature)
    plt.legend(loc='upper left', prop={'size':8}) # bbox_to_anchor=(0, 0, 1.86, 1), 
    plt.savefig(image_output_dirpath + 'predicting_tuning_errors_' + temp_feature + '.pdf') 


########################################################################################
# Now we will assess the expected prediction error using nested cross-validation.
# We will also compare our predictor against a set of baselines.
# The evaluator will prepare identical CV folds and then iterate over various targets.
# Each target is defined by a predictor and the particular input and output fields to use.
########################################################################################

from pysf.generalisation import GeneralisationPerformanceEvaluator
from pysf.predictors.baselines import SeriesMeansPredictor, ZeroPredictor, TimestampMeansPredictor, SeriesLinearInterpolator

evaluator_weather = GeneralisationPerformanceEvaluator(data=data_weather, prediction_times=common_prediction_times)

evaluator_weather.add_to_targets(   combos_of_input_time_column=[ True ], combos_of_input_value_colnames=[ None ], combos_of_output_value_colnames=[ common_prediction_features ]
                                , predictor_templates={   'Baseline single-series series means' : SeriesMeansPredictor()
                                                        , 'Baseline 0 values' : ZeroPredictor()
                                                        , 'Baseline multi-series timestamp means' : TimestampMeansPredictor()
                                                        , 'Baseline single-series series linear interpolator' : SeriesLinearInterpolator()
                                                      })
    
evaluator_weather.add_to_targets(   combos_of_input_time_column=[ False ], combos_of_input_value_colnames=[ common_prediction_features ], combos_of_output_value_colnames=[ common_prediction_features ]
                                , predictor_templates={   'Multi-series self-tuning Smoothing PCR' : tuning_overall_multiseries_smoothing_pcr_predictor })
    
random.seed(777) # for reproducibility
results_df = evaluator_weather.evaluate()

evaluator_weather.chart_overall_performance(feature_name='tempav', metric='rmse')
evaluator_weather.chart_per_timestamp_performance(feature_name='tempav', metric='rmse')

evaluator_weather.chart_overall_performance(feature_name='precav', metric='rmse')
evaluator_weather.chart_per_timestamp_performance(feature_name='precav', metric='rmse')


########################################################################################
# Exclude from dissertation
########################################################################################

if save_images:
    evaluator_weather.chart_overall_performance(feature_name='tempav', metric='rmse')
    plt.tight_layout()
    plt.savefig(image_output_dirpath + 'generalisation_tempav_overall.pdf') 

if save_images:
    evaluator_weather.chart_per_timestamp_performance(feature_name='tempav', metric='rmse')
    plt.legend(loc='upper left', prop={'size':6}) # bbox_to_anchor=(0, 0, 1.86, 1), 
    plt.tight_layout()
    plt.savefig(image_output_dirpath + 'generalisation_tempav_per_timestamp.pdf') 
    
if save_images:
    evaluator_weather.chart_overall_performance(feature_name='precav', metric='rmse')
    plt.tight_layout()
    plt.savefig(image_output_dirpath + 'generalisation_precav_overall.pdf') 

if save_images:
    evaluator_weather.chart_per_timestamp_performance(feature_name='precav', metric='rmse')
    plt.legend(loc='upper left', prop={'size':6}) # bbox_to_anchor=(0, 0, 1.86, 1), 
    plt.tight_layout()
    plt.savefig(image_output_dirpath + 'generalisation_precav_per_timestamp.pdf') 
    



