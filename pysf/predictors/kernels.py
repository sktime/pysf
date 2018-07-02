
from .logger import LoggingHandler 
from .data import MultiSeries
from .framework import MultiCurveTabularPredictor

# See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

import numpy as np



# Converts a variance-covariance matrix into a correlation matrix
def cov_to_corr_matrices(cov_mat):
    sds = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / sds[:, None] / sds[None, :] # this uses numpy's broadcasting
    return corr_mat
    
    
# Run kernel ridge regression for each series (row) in the given matrix
def get_list_of_alphas(a1d_times, scaled_a3d_vs_times, krr_estimator):
    count_series = scaled_a3d_vs_times.shape[0]
    list_series_idx_to_alphas = [None] * count_series
    for series_idx in range(count_series):    
        #print('Running kernel ridge regression for series idx ' + str(series_idx))
        #########################################################################################################
        # for a given series i, KRR of t_i => series values x_i(t_i) using the given kernel as a parameter
        #########################################################################################################
        krr_estimator.fit(X=a1d_times.reshape(-1,1), y=scaled_a3d_vs_times[series_idx, :, :])
        alphas = krr_estimator.dual_coef_
        #print(alphas.shape) # (365, 2)
        list_series_idx_to_alphas[series_idx] = alphas
        #print(np.dot(alphas, train_a3d_vs_times[series_idx, :, :]).shape) doesn't work
    return list_series_idx_to_alphas
    

# Safely convert a general kernel param to a polynomial degree param
def get_polynomial_kernel_params(kernel_param):
    degree_param = abs(int(kernel_param))
    if degree_param < 2:
        degree_param = 2
    gamma_param = None
    coef0_param = 0
    return (degree_param, gamma_param, coef0_param)

    
def evaluate_sklearn_kernel(kernel_name, kernel_param, X, Y=None):
    # These names are consistent with sklearn's
    if kernel_name not in ['linear', 'polynomial', 'rbf', 'laplacian']:
        raise Exception('Unrecognised kernel name \'' + kernel_name + '\'!')
    
    if kernel_name == 'linear':
        return linear_kernel(X=X, Y=Y)
    elif kernel_name == 'polynomial':
        (degree_param, gamma_param, coef0_param) = get_polynomial_kernel_params(kernel_param=kernel_param)
        return polynomial_kernel(X=X, Y=Y, degree=degree_param, gamma=gamma_param, coef0=coef0_param)
    elif kernel_name == 'rbf':
        return rbf_kernel(X=X, Y=Y, gamma=kernel_param)
    else: # Laplacian
        return laplacian_kernel(X=X, Y=Y, gamma=kernel_param)
    

def get_series_kernel(iterated_kernel_name, iterated_kernel_param, row_a1d_times, row_list_series_idx_to_alphas, col_a1d_times, col_list_series_idx_to_alphas):
    count_row_series = len(row_list_series_idx_to_alphas)
    count_col_series = len(col_list_series_idx_to_alphas)

    K = np.zeros((count_row_series, count_col_series))
    for i in range(count_row_series):    
        for j in range(count_col_series):
            #print((i,j))
            #########################################################################################################
            # evaluate a kernel between the timestamps of 2 different series i & j: t_i <=> t_j
            # in our grid-setting, they are both the same
            #########################################################################################################
            k = evaluate_sklearn_kernel(kernel_name=iterated_kernel_name, kernel_param=iterated_kernel_param, X=row_a1d_times.reshape(-1,1), Y=col_a1d_times.reshape(-1,1)) # (365, 365)
            
            alphas_i = row_list_series_idx_to_alphas[i]  # (365, 2)
            alphas_j = col_list_series_idx_to_alphas[j]  # (365, 2)
            K_ij = np.sum(np.dot(np.dot(np.transpose(alphas_i), k), alphas_j)) # convert (2, 2)) to scalar by np.sum
            K[i,j] = K_ij
            #K[j,i] = K_ij
            
            #print("alphas_i" + str(alphas_i))
            #print("alphas_j" + str(alphas_j))
    return K

    
def evaluate_full_series_kernel(X, include_time_as_feature, value_colnames_filter, krr_lambda, krr_kernel_name, krr_kernel_param, iterated_kernel_name, iterated_kernel_param):
    # Extract arrays
    (raw_all_a3d_vs_times, all_a2d_vs_series, all_a1d_times) = X.select_arrays(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter, allow_missing_values=True)
    shape_all_a3d_vs_times = raw_all_a3d_vs_times.shape # cache for later
    #print(shape_all_a3d_vs_times)
    
    count_all_series = shape_all_a3d_vs_times[0]
    count_features = shape_all_a3d_vs_times[2]
    
    # We need to do a preprocessing scaling step. Ensure we keep it fitted to the training data.
    # The scaler needs a 2-D array, so we temporarily reshape each to (#series * # timestamps, #time_features)
    scaler = StandardScaler()
    raw_all_a2d_vs_times = raw_all_a3d_vs_times.reshape(shape_all_a3d_vs_times[0] * shape_all_a3d_vs_times[1], shape_all_a3d_vs_times[2])
    scaled_all_a2d_vs_times = scaler.fit_transform(X=raw_all_a2d_vs_times)
    scaled_all_a3d_vs_times = scaled_all_a2d_vs_times.reshape(shape_all_a3d_vs_times)
    
    # Pass in the KRR params
    if krr_kernel_name == 'polynomial':
        (degree_param, gamma_param, coef0_param) = get_polynomial_kernel_params(kernel_param=krr_kernel_param)
        krr_estimator = KernelRidge(alpha=np.repeat(krr_lambda, count_features), kernel=krr_kernel_name, degree=degree_param, gamma=gamma_param, coef0=coef0_param)
    else:
        krr_estimator = KernelRidge(alpha=np.repeat(krr_lambda, count_features), kernel=krr_kernel_name, gamma=krr_kernel_param)

    alphas_all = get_list_of_alphas(a1d_times=all_a1d_times, scaled_a3d_vs_times=scaled_all_a3d_vs_times, krr_estimator=krr_estimator)
    #print(len(alphas_all))
    
    K_all_vs_all = get_series_kernel(iterated_kernel_name=iterated_kernel_name, iterated_kernel_param=iterated_kernel_param, row_a1d_times=all_a1d_times, row_list_series_idx_to_alphas=alphas_all, col_a1d_times=all_a1d_times, col_list_series_idx_to_alphas=alphas_all)
    #print(K_all_vs_all.shape)
    
    return K_all_vs_all
    

    
    
class MultiCurveKernelsPredictor(MultiCurveTabularPredictor):
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
    ParameterNameKrrLambda = 'krr_lambda'
    ParameterNameKrrKernelName = 'krr_kernel_name'
    ParameterNameKrrKernelParam = 'krr_kernel_param'
    ParameterNameIteratedKernelName = 'iterated_kernel_name'
    ParameterNameIteratedKernelParam = 'iterated_kernel_param'
    ParameterNamePredictionLambda = 'prediction_lambda'
    
    def __init__(self, allow_missing_values=False):
        super(MultiCurveKernelsPredictor, self).__init__(classic_estimator=None, allow_missing_values=allow_missing_values)
        # Parameters
        self.krr_lambda = None
        self.krr_kernel_name = None
        self.krr_kernel_param = None
        self.iterated_kernel_name = None
        self.iterated_kernel_param = None
        self.prediction_lambda = None
    
        
    # Override the implementation of this abstract method
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            raise Exception('Passed a None parameter_dict')   
        self.krr_lambda = parameter_dict[MultiCurveKernelsPredictor.ParameterNameKrrLambda]
        self.krr_kernel_name = parameter_dict[MultiCurveKernelsPredictor.ParameterNameKrrKernelName]
        self.krr_kernel_param = parameter_dict[MultiCurveKernelsPredictor.ParameterNameKrrKernelParam]
        self.iterated_kernel_name = parameter_dict[MultiCurveKernelsPredictor.ParameterNameIteratedKernelName]
        self.iterated_kernel_param = parameter_dict[MultiCurveKernelsPredictor.ParameterNameIteratedKernelParam]
        self.prediction_lambda = parameter_dict[MultiCurveKernelsPredictor.ParameterNamePredictionLambda]
        self.debug('Have set values from the parameter dictionary ' + str(parameter_dict)) 
        
        
    # Implementation of the abstract method. Does nothing.
    def _fitImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # Note that the fit() implementation needs to store the following variables for use later:
        #   - self.scaler_X
        #   - self.scaler_Y
        #   - self.krr_estimator
        #   - self.count_features
        #   - self.train_X_a1d_times
        #   - self.alphas_train
        #   - self.K_train_vs_train
        #   - self.K_reg_inv
        #   - self.scaled_train_Y_a3d_vs_times
        
        # Hyperparameter validation
        if None in [self.krr_lambda, self.krr_kernel_name, self.iterated_kernel_name]:
            raise Exception('The following hyperparams cannot be None: krr_lambda = ' + str(self.krr_lambda) + ', krr_kernel_name = ' + str(self.krr_kernel_name) + ', iterated_kernel_name = ' + str(self.iterated_kernel_name))
        
        # Split the multicurves into a time region for multi-curve fitting and a time region for multi-curve prediction.
        (Y_multiseries, X_multiseries) = X.split_by_times(given_times=prediction_times)
        self.debug('About to fit to X = ' + str(X_multiseries) + ' & Y = ' + str(Y_multiseries))
        
        #######################################
        # In the time region for fitting...
        #######################################
        (raw_train_X_a3d_vs_times, train_X_a2d_vs_series, self.train_X_a1d_times) = X_multiseries.select_arrays(include_time_as_feature=False, value_colnames_filter=prediction_features, allow_missing_values=self._allow_missing_values)
        shape_train_X_a3d_vs_times = raw_train_X_a3d_vs_times.shape
        #print(shape_train_a3d_vs_times)   # (28, 365, 2)
        self.count_features = shape_train_X_a3d_vs_times[2]
        
        # Scale data: the columns of the 2-D array are features, to ensure consistent scaling
        self.scaler_X = StandardScaler()
        raw_train_X_a2d_vs_times = raw_train_X_a3d_vs_times.reshape(shape_train_X_a3d_vs_times[0] * shape_train_X_a3d_vs_times[1], shape_train_X_a3d_vs_times[2])
        scaled_train_X_a2d_vs_times = self.scaler_X.fit_transform(X=raw_train_X_a2d_vs_times)
        scaled_train_X_a3d_vs_times = scaled_train_X_a2d_vs_times.reshape(shape_train_X_a3d_vs_times)

        # Smooth the data using a KRR estimator and retrieve the Dual fitted coefficients alpha
        self.krr_estimator = KernelRidge(alpha=np.repeat(self.krr_lambda, self.count_features), kernel=self.krr_kernel_name, gamma=self.krr_kernel_param)
        self.alphas_train = get_list_of_alphas(a1d_times=self.train_X_a1d_times, scaled_a3d_vs_times = scaled_train_X_a3d_vs_times, krr_estimator=self.krr_estimator)
        #print(len(alphas_train)) # 28

        # Calculate a composite series kernel
        self.K_train_vs_train = get_series_kernel(iterated_kernel_name=self.iterated_kernel_name, iterated_kernel_param=self.iterated_kernel_param, row_a1d_times=self.train_X_a1d_times, row_list_series_idx_to_alphas=self.alphas_train, col_a1d_times=self.train_X_a1d_times, col_list_series_idx_to_alphas=self.alphas_train)
        #print(K_train_vs_train.shape) # (28, 28)
        
        # Cache this computationally-intensive inversion operation for later
        self.K_reg_inv = np.linalg.inv( self.K_train_vs_train + (self.prediction_lambda * np.identity(self.K_train_vs_train.shape[0])) )
        
        
        #######################################
        # In the time region for prediction...
        ######################################
        (raw_train_Y_a3d_vs_times, train_Y_a2d_vs_series, self.train_Y_a1d_times) = Y_multiseries.select_arrays(include_time_as_feature=False, value_colnames_filter=prediction_features, allow_missing_values=self._allow_missing_values)
        shape_train_Y_a3d_vs_times = raw_train_Y_a3d_vs_times.shape
        
        # Scale data & cache it for later
        self.scaler_Y = StandardScaler()
        raw_train_Y_a2d_vs_times = raw_train_Y_a3d_vs_times.reshape(shape_train_Y_a3d_vs_times[0] * shape_train_Y_a3d_vs_times[1], shape_train_Y_a3d_vs_times[2])
        scaled_train_Y_a2d_vs_times = self.scaler_Y.fit_transform(X=raw_train_Y_a2d_vs_times)
        self.scaled_train_Y_a3d_vs_times = scaled_train_Y_a2d_vs_times.reshape(shape_train_Y_a3d_vs_times)
        


        
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):    
        # Split the multicurves into a time region for multi-curve fitting and a time region for multi-curve prediction.
        (other, X_multiseries) = X.split_by_times(given_times=prediction_times)
        self.info('About to predict from X = ' + str(X_multiseries))
        
        #######################################
        # In the time region for fitting...
        #######################################
        (raw_test_a3d_vs_times, test_a2d_vs_series, test_a1d_times) = X_multiseries.select_arrays(include_time_as_feature=False, value_colnames_filter=prediction_features, allow_missing_values=self._allow_missing_values)
        shape_test_a3d_vs_times = raw_test_a3d_vs_times.shape # cache for later
        print(raw_test_a3d_vs_times.shape)   # (7, 365, 2)

        # Scale data
        raw_test_a2d_vs_times = raw_test_a3d_vs_times.reshape(shape_test_a3d_vs_times[0] * shape_test_a3d_vs_times[1], shape_test_a3d_vs_times[2])
        scaled_test_a2d_vs_times = self.scaler_X.transform(X=raw_test_a2d_vs_times)
        scaled_test_a3d_vs_times = scaled_test_a2d_vs_times.reshape(shape_test_a3d_vs_times)

        # Calculate alphas
        alphas_test = get_list_of_alphas(a1d_times=test_a1d_times, scaled_a3d_vs_times = scaled_test_a3d_vs_times, krr_estimator=self.krr_estimator)
        #print(len(alphas_test)) # 28
        
        # Calculate asymmetric/non-square Gram matrix of the series kernel between test & training series
        K_test_vs_train = get_series_kernel(iterated_kernel_name=self.iterated_kernel_name, iterated_kernel_param=self.iterated_kernel_param, row_a1d_times=test_a1d_times, row_list_series_idx_to_alphas=alphas_test, col_a1d_times=self.train_X_a1d_times, col_list_series_idx_to_alphas=self.alphas_train)
        #print(K_test_vs_train.shape)  # (7, 28)
        
        # For efficiency's sake, the matrix inversion has been done in the fitting stage (and cached as self.K_reg_inv).
        # Calculate the "prediction series covariance matrix" (for lack of a better name):
        KKinv = np.dot(K_test_vs_train, self.K_reg_inv)
        
        #######################################
        # In the time region for prediction...
        ######################################
        
        # For a single time point, predict for multiple series at once, iterating over all time points
        scaled_Y_hat_a3d = np.empty([len(alphas_test), len(prediction_times), self.count_features])
        for prediction_time_idx in range(self.scaled_train_Y_a3d_vs_times.shape[1]):
            scaled_Y_hat_a3d[:, prediction_time_idx, :] = np.dot(KKinv, self.scaled_train_Y_a3d_vs_times[:, prediction_time_idx, :])
        
        # Invert the scalings
        shape_scaled_Y_hat_a3d = scaled_Y_hat_a3d.shape
        scaled_Y_hat_a2d = scaled_Y_hat_a3d.reshape(shape_scaled_Y_hat_a3d[0] * shape_scaled_Y_hat_a3d[1], shape_scaled_Y_hat_a3d[2])
        raw_Y_hat_a2d = self.scaler_Y.inverse_transform(X=scaled_Y_hat_a2d)
        raw_Y_hat_a3d = raw_Y_hat_a2d.reshape(shape_scaled_Y_hat_a3d)
        
        # Wrap up the predictions in a MultiSeries
        Y_hat = X.new_instance_from_3d_array(a3d_vs_times=raw_Y_hat_a3d, times=prediction_times, value_colnames_vs_times=prediction_features)
        return Y_hat
        

        
    # Implementation of the abstract method.
    def get_deep_copy(self):
        # (There's no need to copy over a copy of the classic estimator since only 1 instance lives per "fit")
        res = MultiCurveKernelsPredictor(allow_missing_values=self._allow_missing_values)
        # Params
        res.krr_lambda = self.krr_lambda
        res.krr_kernel_name = self.krr_kernel_name
        res.krr_kernel_param = self.krr_kernel_param
        res.iterated_kernel_name = self.iterated_kernel_name
        res.iterated_kernel_param = self.iterated_kernel_param
        res.prediction_lambda = self.prediction_lambda
        return res        
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(krr_lambda = ' + str(self.krr_lambda) +', krr_kernel_name = ' + str(self.krr_kernel_name) + ', krr_kernel_param = ' + str(self.krr_kernel_param) + ', iterated_kernel_name = ' + str(self.iterated_kernel_name) + ', iterated_kernel_param = ' + str(self.iterated_kernel_param) + ', prediction_lambda = ' + str(self.prediction_lambda) + ')')
  

        
        
        
    

##################################################
# For testing
##################################################


###############################
# Multivariate weather data
###############################


if False:
    
    from pysf.data import load_ramsay_weather_data_dfs, load_ramsay_growth_data_dfs, MultiSeries
    from sklearn.model_selection import KFold

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

    
    # This is a slightly hacky way to generate a single training/test split, since my validation prevents you passing in k=1
    splits = list(data_weather.generate_series_folds(series_splitter=KFold(n_splits=5)))
    (training_instance, validation_instance) = splits[0]
    #training_instance.visualise(title='Training instance')
    #validation_instance.visualise(title='Validation instance')
    
    predictor = MultiCurveKernelsPredictor()
    predictor.set_parameters({  'krr_lambda'            : 1.23
                             , 'krr_kernel_name'       : 'rbf'
                             , 'krr_kernel_param'      : 1e-5
                             , 'iterated_kernel_name'  : 'rbf'
                             , 'iterated_kernel_param' : 1e-5
                             , 'prediction_lambda' : 0.789
                             })
    print(predictor)

    
    # Common target
    include_timestamps_as_features = False
    times = np.arange(301,366)
    prediction_features = ['tempav','precav']
    
    predictor.fit(X=training_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    scoring_results = predictor.score(X=validation_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    
    individual_scoring_feature_name = 'tempav'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='MultiCurveKernelsPredictor: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='MultiCurveKernelsPredictor: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='MultiCurveKernelsPredictor, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)
    
    individual_scoring_feature_name = 'precav'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='MultiCurveKernelsPredictor: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='MultiCurveKernelsPredictor: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='MultiCurveKernelsPredictor, predicting for both features at the same time' + '\n' + 'and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)


###############################
# Univariate growth data
###############################

if False:
    
    import pandas as pd
    
    from pysf.data import load_ramsay_weather_data_dfs, load_ramsay_growth_data_dfs, MultiSeries
    from sklearn.model_selection import KFold
    
    # Data: growth
    (growth_vs_times_df, growth_vs_series_df) = load_ramsay_growth_data_dfs()
    growth_vs_series_df['gender'] = growth_vs_series_df['gender'].astype('category')
    growth_vs_series_df = pd.concat([growth_vs_series_df, pd.get_dummies(growth_vs_series_df['gender'])], axis=1)
    data_growth = MultiSeries(data_vs_times_df=growth_vs_times_df, data_vs_series_df=growth_vs_series_df, time_colname='age', series_id_colnames=['gender', 'cohort_id'])
    #data_growth.visualise()

    # This is a slightly hacky way to generate a single training/test split, since my validation prevents you passing in k=1
    splits = list(data_growth.generate_series_folds(series_splitter=KFold(n_splits=5)))
    (training_instance, validation_instance) = splits[0]
    #training_instance.visualise(title='Training instance')
    #validation_instance.visualise(title='Validation instance')
    
    predictor = MultiCurveKernelsPredictor()
    predictor.set_parameters({  'krr_lambda'            : 1.23
                             , 'krr_kernel_name'       : 'rbf'
                             , 'krr_kernel_param'      : 1e-5
                             , 'iterated_kernel_name'  : 'rbf'
                             , 'iterated_kernel_param' : 1e-5
                             , 'prediction_lambda' : 0.789
                             })
    print(predictor)

    
    # Common target
    include_timestamps_as_features = False
    times = np.arange(14,18.5,0.5)
    prediction_features = ['height']
    
    predictor.fit(X=training_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    scoring_results = predictor.score(X=validation_instance, input_time_feature=include_timestamps_as_features, prediction_times=times, prediction_features=prediction_features)
    
    individual_scoring_feature_name = 'height'
    individual_result = scoring_results[individual_scoring_feature_name]
    individual_result.Y_true.visualise(title='MultiCurveKernelsPredictor: Y_true',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.Y_hat.visualise(title='MultiCurveKernelsPredictor: Y_hat',  filter_value_colnames=individual_scoring_feature_name)
    individual_result.residuals.visualise()
    title='MultiCurveKernelsPredictor, predicting and scoring for ' + individual_scoring_feature_name +' only'
    individual_result.err.visualise_per_timestamp(title=title)
    individual_result.err.visualise_overall(title=title)
    
