

from .logger import LoggingHandler 
from .data import MultiSeries
from .errors import RawResiduals, ErrorCurve
from .transformers.framework import AbstractTransformer

import numpy as np
from abc import ABC, abstractmethod
import copy




# Simple container to associate actual & predicted values with the calculated error curve and raw residuals.
class ScoringResult(LoggingHandler):
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
    def __init__(self, feature_name, Y_true, Y_hat, residuals, err):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        super(ScoringResult, self).__init__()
        if type(feature_name) != str:
            raise Exception('Was expecting a str object for param feature_name! Instead was initialised with a ' + str(type(feature_name)))
        if type(Y_true) != MultiSeries:
            raise Exception('Was expecting a MultiSeries object for param Y_true! Instead was initialised with a ' + str(type(Y_true)))
        if type(Y_hat) != MultiSeries:
            raise Exception('Was expecting a MultiSeries object for param Y_hat! Instead was initialised with a ' + str(type(Y_hat)))
        if type(residuals) != RawResiduals:
            raise Exception('Was expecting a RawResiduals object for param residuals! Instead was initialised with a ' + str(type(residuals)))
        if type(err) != ErrorCurve:
            raise Exception('Was expecting an ErrorCurve object for param err! Instead was initialised with a ' + str(type(err)))
        self.feature_name = feature_name
        self.residuals = residuals
        self.err = err
        self.Y_true = Y_true
        self.Y_hat = Y_hat
        
    # For serialization via Pickle
    def __getstate__(self):
        state_dict = {}
        state_dict['feature_name'] = self.feature_name
        state_dict['residuals'] = self.residuals
        state_dict['err'] = self.err
        return state_dict
        
    # For deserialization via Pickle
    def __setstate__(self, state):
        state_dict = state
        self.feature_name = state_dict['feature_name']
        self.residuals = state_dict['residuals']
        self.err = state_dict['err']
        self.initLogger()
        
        

class AbstractPredictor(ABC, LoggingHandler):
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
        super(AbstractPredictor, self).__init__()
        self._isFitted = False

    @abstractmethod
    def set_parameters(self, parameter_dict):      
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """    
        pass
        
    def _setClassicEstimatorParams(self, classic_estimator, parameter_dict):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        if classic_estimator is None or parameter_dict is None or len(parameter_dict) == 0:
            self.debug('Will not attempt to set any parameters: classic_estimator = ' + str(classic_estimator) + ', parameter_dict = ' + str(parameter_dict))
        else:
            valid_keys = list(classic_estimator.get_params().keys())
            given_keys = list(parameter_dict.keys())
            invalid_keys = np.setdiff1d(given_keys, valid_keys)
            parameter_dict = dict(parameter_dict)
            for key in invalid_keys:
                del parameter_dict[key]
            classic_estimator.set_params(**parameter_dict) # The double-asterisk syntax tranforms the dict into (the required format of) keyword args
    
    def _enforceTypeMultiSeries(self, paramValue, paramName):
        if type(paramValue) != MultiSeries:
            raise Exception('Expected ' + paramName + ' to be a MultiSeries but instead it was a ' + str(type(paramValue)))
            
    def _enforceTimesShape(self, times):
        if (len(times.shape) != 1):
            raise Exception('Expected the times parameter to be a 1D array but instead it had shape ' + str(times.shape))

    # X must be a MultiSeries
    def fit(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        self._enforceTypeMultiSeries(paramValue=X, paramName='X')
        self._fitImplementation(X=X, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_times=prediction_times, prediction_features=prediction_features)
        self._isFitted = True
        
    @abstractmethod
    def _fitImplementation(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):   
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """       
        pass
        
    # Returns a MultiSeries
    # X must be a MultiSeries
    # If prediction_features is specified, those named features will be predicted/scored, otherwise all applicable features will be.
    def predict(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        self._enforceTypeMultiSeries(paramValue=X, paramName='X')
        self._enforceTimesShape(times=prediction_times)
        if self._isFitted:
            return self._predictImplementation(X=X, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_times=prediction_times, prediction_features=prediction_features)
        else:
            raise Exception('This instance has not been fitted yet. You need to fit any AbstractPredictor implementation before you can use it to predict!')
        
    @abstractmethod
    def _predictImplementation(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):  
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """          
        pass
        
    # X must be a MultiSeries
    # If prediction_features is specified, those named features will be predicted/scored individually, otherwise all applicable features will be.
    # Return a ScoreResult.
    def score(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # Validation
        if type(prediction_features) == str:
            prediction_features = [prediction_features] # required
        self._enforceTypeMultiSeries(paramValue=X, paramName='X')
        self._enforceTimesShape(times=prediction_times)
        
        # Do the prediction
        (Y_true, X_input) = X.split_by_times(given_times=prediction_times)
        Y_hat = self.predict(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        
        total_obs_Y_true = Y_true.count_observations
        total_obs_Y_hat = Y_hat.count_observations
        if total_obs_Y_true != total_obs_Y_hat:
            self.warning('Y_true has ' + str(total_obs_Y_true) + ' total observations but Y_hat has ' + str(total_obs_Y_hat) + ' total observations.')
        if total_obs_Y_hat == 0:
            self.error('Y_hat = ' + str(Y_hat))
            self.error('Y_hat._data_vs_times_df = ' + str(Y_hat._data_vs_times_df))
            raise Exception('No observations in Y_hat!')
            
        # Return a ScoringResult for each individual feature, collected into a dict
        dict_of_results = {}
        for prediction_feature in prediction_features:
            # Calculate the error curve from the residuals, and return them together with MultiSeries for convenience.
            residuals = Y_true.get_raw_residuals(Y_hat=Y_hat, value_colnames_vs_times_filter=prediction_feature)
            err = ErrorCurve.init_from_raw_residuals(raw_residuals_obj=residuals)
            individual_result = ScoringResult(feature_name=prediction_feature, Y_true=Y_true, Y_hat=Y_hat, residuals=residuals, err=err)
            dict_of_results[prediction_feature] = individual_result
        return dict_of_results

    # Non-abstract method that by default does nothing. This will help us reduce memory leakage when performing large-scale experiments.    
    def compact(self):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        pass
        
    # Make sure this is implemented at the very bottom of the inheritance tree! 
    # The reason we do this instead of copy.deepcopy is that the alternative fails, probably 
    # due to an implicit lock/mutex object buried within the logging package.
    @abstractmethod
    def get_deep_copy(self):          
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        pass
        
    

class PipelinePredictor(AbstractPredictor): 
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
    def __init__(self, chain = []):  
        super(PipelinePredictor, self).__init__()
        if type(chain) != list:
            chain = [ chain ]
        self._chain = chain
        
    # Implementation of the abstract method.
    def set_parameters(self, parameter_dict):          
        for elem in self._chain:
            self.debug('Setting parameters ' + str(parameter_dict) + ' on ' + str(elem))
            elem.set_parameters(parameter_dict = parameter_dict)

    # Implementation of the abstract method.
    def _fitImplementation(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):          
        idx = 0
        for elem in self._chain:
            idx = idx + 1
            self.debug('Fitting Element ' + str(idx) + '/' + str(len(self._chain)))
            if issubclass(type(elem), AbstractTransformer):
                X = elem.transform(X=X)
            elif issubclass(type(elem), AbstractPredictor):
                elem.fit(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
            else:
                raise Exception('Unrecognised type ' + str(type(elem)) + ' of ' + str(elem))
        
    # Implementation of the abstract method.
    def _predictImplementation(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):            
        idx = 0
        for elem in self._chain:
            idx = idx + 1
            self.debug('Predicting from Element ' + str(idx) + '/' + str(len(self._chain)))
            if issubclass(type(elem), AbstractTransformer):
                X = elem.transform(X=X)
            elif issubclass(type(elem), AbstractPredictor):
                X= elem.predict(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
            else:
                raise Exception('Unrecognised type ' + str(type(elem)) + ' of ' + str(elem))
        return X
        
    # Override of non-abstract method
    def compact(self):
        for elem in self._chain:
            if issubclass(type(elem), AbstractPredictor):
                elem.compact()
            
    # Implementation of the abstract method.
    def get_deep_copy(self):
        return PipelinePredictor(chain = [elem.get_deep_copy() for elem in self._chain])
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '( ' + ' ==> '.join([str(elem) for elem in self._chain]) + ' )')
        

    
# This works only with SKLearn Estimators conforming to the usual fit/predict pattern at the moment. 
# If it should be extended, then consider adding some more layers of abstraction/inheritance.
class SingleCurveSeriesPredictor(AbstractPredictor):
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
    def __init__(self, classic_estimator, allow_missing_values=False):
        super(SingleCurveSeriesPredictor, self).__init__()
        self._classic_estimator = classic_estimator
        self._allow_missing_values = allow_missing_values
        
        
    def _enforceAtLeastOneInputFeature(self, input_time_feature, input_non_time_features):
        if not(input_time_feature) and (input_non_time_features is None or len(input_non_time_features) == 0):
            raise Exception('Need at least 1 input (covariate)! input_time_feature = ' + str(input_time_feature) + ', input_non_time_features = ' + str(input_non_time_features))
    
            
    # Implementation of the abstract method. Simply passes the parameters through to the underlying sklearn estimator.
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            self.debug('Passed a None parameter_dict')
        else:
            self.debug('Passing through the given parameter_dict ' + str(parameter_dict) + ' to the underlying sklearn estimator ' + str(self._classic_estimator))
            self._setClassicEstimatorParams(classic_estimator=self._classic_estimator, parameter_dict=parameter_dict)
            
        
    # Implementation of the abstract method. Does nothing.
    def _fitImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        #self._enforceAtLeastOneInputFeature(input_time_feature=input_time_feature, input_non_time_features=input_non_time_features)
        self.debug('Fit: do nothing')
        
        
    # Overrideable method to call fit() on the classic estimator
    def _innerClassicFit(self, a2d_X_fit, a2d_Y_fit, prediction_features):
        self.debug('About to call fit on the inner ' + str(self._classic_estimator) + ' with the X & Y arrays having the shapes ' + str(a2d_X_fit.shape) + ' & ' + str(a2d_Y_fit.shape) + ' respectively.')
        self._classic_estimator.fit(a2d_X_fit, a2d_Y_fit)
          
                          
    # Overrideable method to call predict() on the classic estimator
    def _innerClassicPredict(self, a2d_X_predict, prediction_time_start_idx, prediction_time_end_idx):
        self.debug('About to call predict on the inner ' + str(self._classic_estimator) + ' with the X array having the shape ' + str(a2d_X_predict.shape))
        current_Y_hat = self._classic_estimator.predict(a2d_X_predict) 
        self.debug('Return value from calling predict on the inner ' + str(self._classic_estimator) + ' has the shape ' + str(current_Y_hat.shape))
        return current_Y_hat
        
        
    # Implementation of the abstract method
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # Validation
        self._enforceAtLeastOneInputFeature(input_time_feature=input_time_feature, input_non_time_features=input_non_time_features)
        
        # Split the multicurves into a time region for single-curve fitting and a time region for single-curve prediction.
        (multicurves_for_singlecurve_predict, multicurves_for_singlecurve_fit) = X.split_by_times(given_times=prediction_times)
        
        # This is to help out ARIMA prediction, which relies on indices only. Assumes we're predicting forward, and not outside the whole range of X here:
        (prediction_time_start_idx, prediction_time_end_idx) = X.get_time_boundary_indices(given_times=prediction_times)
        
        # Each of these 3-D arrays has the shape (# series, # timestamps, # features)
        if input_non_time_features is None:
            input_non_time_features = []
        (a3d_X_fit, train_times)  = multicurves_for_singlecurve_fit.select_merged_3d_array(     include_time_as_feature=input_time_feature, allow_missing_values=self._allow_missing_values, value_colnames_filter=input_non_time_features)
        (a3d_Y_fit, train_times)  = multicurves_for_singlecurve_fit.select_merged_3d_array(     include_time_as_feature=False,              allow_missing_values=self._allow_missing_values, value_colnames_filter=prediction_features)
        (a3d_X_predict, times)    = multicurves_for_singlecurve_predict.select_merged_3d_array( include_time_as_feature=input_time_feature, allow_missing_values=self._allow_missing_values, value_colnames_filter=input_non_time_features)
        self.debug('Within _predictImplementation(), input array shapes are as follows. For a3d_X_fit: ' + str(a3d_X_fit.shape) + ', a3d_Y_fit: ' + str(a3d_Y_fit.shape) + ', prediction_times: ' + str(prediction_times.shape) + ', a3d_X_predict: ' + str(a3d_X_predict.shape))
        
        # Define a 3-D array of shape (# series, # timestamps, # features) to collect the results in. 
        count_series = a3d_X_fit.shape[0] # assume the # series is identical for all of the data arrays
        count_times = len(prediction_times)
        if prediction_features is None:
            count_prediction_features = X.count_features
        elif type(prediction_features) == str:
            count_prediction_features = 1
        else:
            count_prediction_features = len(prediction_features)
        Y_hat_arr = np.empty([count_series, count_times, count_prediction_features])
        self.debug('Within _predictImplementation(), output array shape is as follows. For Y_hat_arr: ' + str(Y_hat_arr.shape))
        
        # Iterate through all curves, performing the internal SKLearn estimator's fit-predict on every single one.
        # Sklearn's requirement of (n_samples, n_features) corresponds to our (#timestamps, #features), where #features may include the timestamp column
        for series_idx in range(count_series):
            a2d_X_fit     = a3d_X_fit[series_idx,:,:]
            a2d_Y_fit     = a3d_Y_fit[series_idx,:,:]
            a2d_X_predict = a3d_X_predict[series_idx,:,:]
        
            # Call fit() and then predict() on the classic estimator for this single curve
            self._innerClassicFit(a2d_X_fit=a2d_X_fit, a2d_Y_fit=a2d_Y_fit, prediction_features=prediction_features)
            current_Y_hat = self._innerClassicPredict(a2d_X_predict=a2d_X_predict, prediction_time_start_idx=prediction_time_start_idx, prediction_time_end_idx=prediction_time_end_idx)
            
            if len(current_Y_hat.shape) == 1:
                current_Y_hat = current_Y_hat.reshape(-1, 1) # reshaping to have a single feature, SKLearn-style
            Y_hat_arr[series_idx,:,:] = current_Y_hat

        # Convert result from 3D array to MultiSeries object
        Y_hat = multicurves_for_singlecurve_predict.new_instance_from_3d_array(a3d_vs_times=Y_hat_arr, times=prediction_times, value_colnames_vs_times=prediction_features)
        return Y_hat

        
    # Implementation of the abstract method.
    def get_deep_copy(self):
        return SingleCurveSeriesPredictor(classic_estimator=copy.deepcopy(self._classic_estimator), allow_missing_values=self._allow_missing_values)
       
    
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(_classic_estimator = ' + str(self._classic_estimator) + ', allow_missing_values = ' + str(self._allow_missing_values) + ')')
    
        
        
class AbstractMultiCurvePredictor(AbstractPredictor):
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
        super(AbstractMultiCurvePredictor, self).__init__()
        
    def _checkAndCompleteInputNonTimeFeatures(self, input_non_time_features, prediction_features):
        # As well as any specifically-named features (if any) it should include the prediction features, in a union.
        if input_non_time_features is None:
            input_non_time_features = []
        elif type(input_non_time_features) == str:
            input_non_time_features = [input_non_time_features]
        
        # It would be more efficient to do this by union'ing set operations, but operations like this should be deterministic in general:
        for p in prediction_features:
            if p not in input_non_time_features:
                input_non_time_features.append(p)
        
        self.debug('Determined input_non_time_features = ' + str(input_non_time_features))
        return input_non_time_features
        
   
        
class AbstractWindowedPredictor(AbstractPredictor):
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
    ParameterNameInputWindowWidth = 'width_input'
    ParameterNameOutputWindowWidth = 'width_output'
    
    def __init__(self):
        super(AbstractWindowedPredictor, self).__init__()
    
    # Implementation of the abstract method. Simply passes the parameters through to the underlying sklearn estimator.
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            self.debug('Passed a None parameter_dict')
        else:
            # Copy the (mutable) dict so we can remove keys before passing the params through
            parameter_dict = dict(parameter_dict)
            if AbstractWindowedPredictor.ParameterNameInputWindowWidth in parameter_dict:
                self.width_input = int(parameter_dict[AbstractWindowedPredictor.ParameterNameInputWindowWidth]) # explicitly cast to int
                self.debug('Set self.width_input = ' + str(self.width_input))
                del parameter_dict[AbstractWindowedPredictor.ParameterNameInputWindowWidth]
            if AbstractWindowedPredictor.ParameterNameOutputWindowWidth in parameter_dict:
                self.width_output = int(parameter_dict[AbstractWindowedPredictor.ParameterNameOutputWindowWidth]) # explicitly cast to int
                self.debug('Set self.width_output = ' + str(self.width_output))
                del parameter_dict[AbstractWindowedPredictor.ParameterNameOutputWindowWidth]
            if len(parameter_dict) == 0:
                self.debug('No remaining items in the updated parameter_dict, so no need to pass it to the inner SKlearn estimator')
            else:
                self.debug('Passing through the updated parameter_dict ' + str(parameter_dict) + ' to the underlying classic estimator ' + str(self._classic_estimator))
                self._setClassicEstimatorParams(classic_estimator=self._classic_estimator, parameter_dict=parameter_dict)
                
    def _checkWidthsAreSet(self):
        if self.width_input is None or self.width_output is None:
            raise Exception('Both width_input and width_output parameters need to be set on ' + str(self))
    
    def get_forward_sliding_windows(self, arr, max_window_size):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        idx = 0
        sz = arr.size
        while (idx + max_window_size) < sz:
            sub_indices = np.arange(idx, idx + max_window_size)
            sub_vals = arr[sub_indices]
            yield sub_vals
            idx = idx + max_window_size
        if idx < sz:
            sub_indices = np.arange(idx, sz)
            sub_vals = arr[sub_indices]
            self.debug('Yielding an irregularly-shaped window with size ' + str(sub_vals.shape) + ' and values ' + str(sub_vals))
            yield sub_vals
            
    #for s in get_forward_sliding_windows(arr=(np.arange(52)), max_window_size=7):
    #    print(s)
    #
    #for s in get_forward_sliding_windows(arr=np.arange(301,366), max_window_size=2):
    #    print(s)       
        
    
# This works only with SKLearn Estimators conforming to the usual fit/predict pattern at the moment. 
class MultiCurveTabularPredictor(AbstractMultiCurvePredictor):
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
    def __init__(self, classic_estimator, allow_missing_values=False):
        super(MultiCurveTabularPredictor, self).__init__()
        self._classic_estimator = classic_estimator
        self._allow_missing_values = allow_missing_values
        
    # Implementation of the abstract method. Simply passes the parameters through to the underlying sklearn estimator.
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            self.debug('Passed a None parameter_dict')
        else:
            self.debug('Passing through the given parameter_dict ' + str(parameter_dict) + ' to the underlying sklearn estimator ' + str(self._classic_estimator))
            self._setClassicEstimatorParams(classic_estimator=self._classic_estimator, parameter_dict=parameter_dict)
            
    def _selectTabularArrayForFittingOrPredicting(self, X, include_time_as_feature, value_colnames_filter):
        (X_a2d, t) = X.select_tabular_full_2d_array(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter) 
        return X_a2d
            
    def _fitClassicEstimator(self, X_arr, Y_arr):
        # Fit the sklearn estimator with X in the shape of [n_samples,n_features] taking in our (# series, # times * # features)
        self.debug('About to call fit on the inner ' + str(self._classic_estimator) + ' with the X & Y arrays having the shapes ' + str(X_arr.shape) + ' & ' + str(Y_arr.shape) + ' respectively.')
        self._classic_estimator.fit(X=X_arr, y=Y_arr) 
                
    # Implementation of the abstract method
    def _fitImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # Validation
        if prediction_features is None:
            raise Exception('Must specify prediction_features!')
        if type(prediction_features) == str:
            prediction_features = [prediction_features] # necessary
        input_non_time_features = self._checkAndCompleteInputNonTimeFeatures(input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        
        # Split the multicurves into a time region for multi-curve fitting and a time region for multi-curve prediction.
        (Y_multiseries, X_multiseries) = X.split_by_times(given_times=prediction_times)
        self.debug('About to fit to X = ' + str(X_multiseries) + ' & Y = ' + str(Y_multiseries))
        
        # Convert (# series, # times, # features) to (#series, #times * #features) for both the X & Y time regions:
        X_arr = self._selectTabularArrayForFittingOrPredicting(X=X_multiseries, include_time_as_feature=input_time_feature,  value_colnames_filter=input_non_time_features) 
        Y_arr = self._selectTabularArrayForFittingOrPredicting(X=Y_multiseries, include_time_as_feature=False,               value_colnames_filter=prediction_features) 
        
        self._fitClassicEstimator(X_arr=X_arr, Y_arr=Y_arr)
    
    def _predictFromClassicEstimator(self, X_arr):
        # Do the multivariate prediction
        self.debug('About to call predict on the inner ' + str(self._classic_estimator) + ' with the X array having the shape ' + str(X_arr.shape))
        Y_hat_a2d = self._classic_estimator.predict(X=X_arr) 
        return Y_hat_a2d
        
    # Implementation of the abstract method
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # Validation: check & update input_non_time_features
        if prediction_features is None:
            raise Exception('Must specify prediction_features!')
        if type(prediction_features) == str:
            prediction_features = [prediction_features] # necessary
        input_non_time_features = self._checkAndCompleteInputNonTimeFeatures(input_non_time_features=input_non_time_features, prediction_features=prediction_features)
            
        # Split the multicurves into a time region for multi-curve fitting and a time region for multi-curve prediction.
        (other, X_multiseries) = X.split_by_times(given_times=prediction_times)
        self.debug('About to predict from X = ' + str(X_multiseries))
        
        # Convert (# series, # times, # features) to (#series, #times * #features) for the X time region only:
        X_arr = self._selectTabularArrayForFittingOrPredicting(X=X_multiseries, include_time_as_feature=input_time_feature,  value_colnames_filter=input_non_time_features) 
        
        Y_hat_a2d = self._predictFromClassicEstimator(X_arr=X_arr)
        if len(Y_hat_a2d.shape) == 1:
            # Possible to get back a simple array, thanks to sklearn!
            Y_hat_a2d = Y_hat_a2d.reshape(X_arr.shape[0], int(len(Y_hat_a2d) / X_arr.shape[0]))
        
        # Convert the result to a MultiSeries
        Y_hat = X.new_instance_from_2d_array(Y_a2d=Y_hat_a2d, times=prediction_times, prediction_features=prediction_features)
        return Y_hat
    
    # Implementation of the abstract method.
    def get_deep_copy(self):
        return MultiCurveTabularPredictor(classic_estimator=copy.deepcopy(self._classic_estimator), allow_missing_values=self._allow_missing_values)
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(_classic_estimator = ' + str(self._classic_estimator) + ', allow_missing_values = ' + str(self._allow_missing_values) + ')')
    

class MultiCurveTabularWindowedPredictor(AbstractMultiCurvePredictor, AbstractWindowedPredictor):
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
    ParameterNameTrainOverPredictionTimes = 'train_over_prediction_times'

    def __init__(self, classic_estimator, allow_missing_values=False):
        super(MultiCurveTabularWindowedPredictor, self).__init__()
        self._classic_estimator = classic_estimator
        self._allow_missing_values = allow_missing_values
        # Need to initialise to None to avoid breaking the deep copy while tuning an unparameterised instance
        self.width_input = None
        self.width_output = None
        self.train_over_prediction_times = None

    @staticmethod
    def initWithAllParams(classic_estimator, allow_missing_values, width_input, width_output, train_over_prediction_times):
        res = MultiCurveTabularWindowedPredictor(classic_estimator=classic_estimator, allow_missing_values=allow_missing_values)
        res.width_input = int(width_input)       # explicitly cast to int
        res.width_output = int(width_output)     # explicitly cast to int
        res.train_over_prediction_times = train_over_prediction_times
        return res
        
    # Implementation of the abstract method. Simply passes the parameters through to the underlying sklearn estimator.
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            self.debug('Passed a None parameter_dict')
        else:
            # Copy the (mutable) dict so we can remove keys before passing the params through
            parameter_dict = dict(parameter_dict)
            if MultiCurveTabularWindowedPredictor.ParameterNameTrainOverPredictionTimes in parameter_dict:
                self.train_over_prediction_times = parameter_dict[MultiCurveTabularWindowedPredictor.ParameterNameTrainOverPredictionTimes]
                self.debug('Set self.train_over_prediction_times = ' + str(self.train_over_prediction_times))
                del parameter_dict[MultiCurveTabularWindowedPredictor.ParameterNameTrainOverPredictionTimes]
            if len(parameter_dict) == 0:
                self.debug('No remaining items in the updated parameter_dict, so no need to pass it up a level to the AbstractWindowedPredictor')
            else:
                self.debug('Passing up the updated parameter_dict ' + str(parameter_dict) + ' to the overridden method in the parent AbstractWindowedPredictor.')
                AbstractWindowedPredictor.set_parameters(self, parameter_dict) # explicit-name version of super()
        
    def _selectTabularWindowedArraysForFitting(self, X, include_time_as_feature, value_colnames_filter):
        (X_arr, Y_arr) = X.select_paired_tabular_windowed_2d_arrays(include_time_as_feature=include_time_as_feature,  value_colnames_filter=value_colnames_filter,  allow_missing_values=self._allow_missing_values,  input_sliding_window_size=self.width_input, output_sliding_window_size=self.width_output)
        return (X_arr, Y_arr)
                
    def _fitClassicEstimator(self, X_arr, Y_arr):
        # Fit the sklearn estimator with X in the shape of [n_samples,n_features] taking in our (# series, # times * # features)
        self.debug('About to call fit on the inner ' + str(self._classic_estimator) + ' with the X & Y arrays having the shapes ' + str(X_arr.shape) + ' & ' + str(Y_arr.shape) + ' respectively.')
        self._classic_estimator.fit(X=X_arr, y=Y_arr) 
        
    # Implementation of the abstract method
    def _fitImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # Validation
        if prediction_features is None:
            raise Exception('Must specify prediction_features!')
        if type(prediction_features) == str:
            prediction_features = [prediction_features] # necessary
        input_non_time_features = self._checkAndCompleteInputNonTimeFeatures(input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        self._checkWidthsAreSet()
        
        # Split the multicurves into a time region for multi-curve fitting and a time region for multi-curve prediction.
        if self.train_over_prediction_times:
            X_train = X
        else:
            (X_prediction_times, X_other_times) = X.split_by_times(given_times=prediction_times)
            X_train = X_other_times
        self.debug('About to fit to X_train = ' + str(X_train))
        
        # Convert (# series, # times, # features) to (#series * #splits, window_size * #time_features + #series_features) for both the X & Y time regions:
        (X_arr, other) = self._selectTabularWindowedArraysForFitting(X=X_train, include_time_as_feature=input_time_feature,  value_colnames_filter=input_non_time_features  )
        (other, Y_arr) = self._selectTabularWindowedArraysForFitting(X=X_train, include_time_as_feature=False,               value_colnames_filter=prediction_features      )
        
        self._fitClassicEstimator(X_arr=X_arr, Y_arr=Y_arr)
    
    def _selectTabularArrayForPredicting(self, X, include_time_as_feature, value_colnames_filter):
        (X_a2d, t) = X.select_tabular_full_2d_array(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter) 
        return X_a2d
     
    def _predictFromClassicEstimator(self, X_arr):
        # Do the multivariate prediction
        self.debug('About to call predict on the inner ' + str(self._classic_estimator) + ' with the X array having the shape ' + str(X_arr.shape))
        Y_hat_a2d = self._classic_estimator.predict(X=X_arr) 
        return Y_hat_a2d
    
    # Implementation of the abstract method
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # Validation: check & update input_non_time_features
        if prediction_features is None:
            raise Exception('Must specify prediction_features!')
        if type(prediction_features) == str:
            prediction_features = [prediction_features] # necessary
        input_non_time_features = self._checkAndCompleteInputNonTimeFeatures(input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        self._checkWidthsAreSet()
         
        # Split the multicurves into a time region for multi-curve fitting and a time region for multi-curve prediction.
        (other, X_input) = X.split_by_times(given_times=prediction_times)
        self.debug('About to predict from X = ' + str(X_input))
        
        # Take a copy and re-index if necessary
        Y_hat = X_input.new_mutable_instance(filter_value_colnames_vs_times=prediction_features, prediction_times=prediction_times)
            
        for prediction_times_subset in self.get_forward_sliding_windows(arr=prediction_times, max_window_size=self.width_output):
            # Combine this with predictions
            X_window = Y_hat.get_backward_time_window(self.width_input)
            
            # Convert (# series, # times, # features) to (#series, #times * #features) for the X time region only:
            X_arr = self._selectTabularArrayForPredicting(X=X_window, include_time_as_feature=input_time_feature, value_colnames_filter=input_non_time_features)
            
            Y_hat_a2d = self._predictFromClassicEstimator(X_arr=X_arr) 
            
            # Reshape to a 3D array. From (#series, #timestamps * # features)
            # to (#series, #timestamps, #features)
            Y_hat_a3d = Y_hat_a2d.reshape(Y_hat_a2d.shape[0], self.width_output, len(prediction_features))
            
            # Take a subset of the predicted timestamps to maintain them within the prediction_times range
            Y_hat_a3d = Y_hat_a3d[:, np.arange(len(prediction_times_subset)), :]
            
            # Concatenate X_window + Y_hat_subset
            Y_hat.update_from_3d_array(a3d_vs_times=Y_hat_a3d, times=prediction_times_subset, value_colnames_vs_times=prediction_features)
            #Y_hat_subset = X_window.new_instance_from_3d_array(a3d_vs_times=Y_hat_a3d, times=prediction_times_subset, value_colnames_vs_times=prediction_features)
            #Y_hat.append(Y_hat_subset)
            
        # END for prediction_times_subset
        
        # Filter by times, if necessary, in case we have predicted too far into the future
        (Y_hat_filtered, other)  = Y_hat.split_by_times(given_times=prediction_times)
        
        if Y_hat_filtered.count_observations == 0:
            Y_hat_filtered.visualise(title='!!! Y_hat_filtered !!!')
            other.visualise(title='!!! other !!!')
            raise Exception('No observations in returned Y_hat_filtered ' + str(Y_hat_filtered))
        
        return Y_hat_filtered
     
    # Implementation of the abstract method.
    def get_deep_copy(self):
        res = MultiCurveTabularWindowedPredictor(classic_estimator=copy.deepcopy(self._classic_estimator), allow_missing_values=self._allow_missing_values)
        res.width_input = self.width_input
        res.width_output = self.width_output
        res.train_over_prediction_times = self.train_over_prediction_times
        return res
        
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(width_input = ' + str(self.width_input) +', width_output = ' + str(self.width_output) + ', _classic_estimator = ' + str(self._classic_estimator) + ', allow_missing_values = ' + str(self._allow_missing_values) + ')')
         

        
        
class SingleCurveTabularWindowedPredictor(AbstractMultiCurvePredictor, AbstractWindowedPredictor):
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
    def __init__(self, classic_estimator, allow_missing_values=False):
        super(SingleCurveTabularWindowedPredictor, self).__init__()
        self._classic_estimator = classic_estimator
        self._allow_missing_values = allow_missing_values
        # Need to initialise to None to avoid breaking the deep copy while tuning an unparameterised instance
        self.width_input = None
        self.width_output = None

    @staticmethod
    def initWithAllParams(classic_estimator, allow_missing_values, width_input, width_output):
        res = SingleCurveTabularWindowedPredictor(classic_estimator=classic_estimator, allow_missing_values=allow_missing_values)
        res.width_input = int(width_input)       # explicitly cast to int
        res.width_output = int(width_output)     # explicitly cast to int
        return res
        
    # Implementation of the abstract method
    def _fitImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        self.debug('Fit: do nothing')
    
    # The first dim. must be # series
    def _selectTabularWindowedArraysForFitting(self, X, include_time_as_feature, value_colnames_filter):
        (X_arr, Y_arr) = X.select_paired_tabular_windowed_3d_by_series_arrays(include_time_as_feature=include_time_as_feature,  value_colnames_filter=value_colnames_filter,  allow_missing_values=self._allow_missing_values,  input_sliding_window_size=self.width_input, output_sliding_window_size=self.width_output)
        return (X_arr, Y_arr)
                    
    # The first dim. must be # series
    def _selectTabularArrayForPredicting(self, X, include_time_as_feature, value_colnames_filter):
        (X_a2d, t) = X.select_tabular_full_2d_array(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter) 
        return X_a2d
        
    # Implementation of the abstract method
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        if prediction_features is None:
            raise Exception('Must specify prediction_features!')
        if type(prediction_features) == str:
            prediction_features = [prediction_features] # necessary
        input_non_time_features = self._checkAndCompleteInputNonTimeFeatures(input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        self._checkWidthsAreSet()
        
        # Split the multicurves into a time region for single-curve fitting and a time region for single-curve prediction.
        (multicurves_for_singlecurve_predict, multicurves_for_singlecurve_fit) = X.split_by_times(given_times=prediction_times)
        
        # Convert (# series, # times, # features) to (#series, #splits, window_size * #time_features + #series_features) for both the X & Y time regions:
        (arr_X_fit, other) = self._selectTabularWindowedArraysForFitting(X=multicurves_for_singlecurve_fit, include_time_as_feature=input_time_feature,  value_colnames_filter=input_non_time_features)
        (other, arr_Y_fit) = self._selectTabularWindowedArraysForFitting(X=multicurves_for_singlecurve_fit, include_time_as_feature=False,               value_colnames_filter=prediction_features)
        self.debug('Within _predictImplementation(), input array shapes are as follows. For arr_X_fit: ' + str(arr_X_fit.shape) + ', arr_Y_fit: ' + str(arr_Y_fit.shape) + ', prediction_times: ' + str(prediction_times.shape))
        
        # Take a copy and re-index if necessary
        self.info('Will predict one-by-one from X = ' + str(multicurves_for_singlecurve_fit))
        Y_hat = multicurves_for_singlecurve_fit.new_mutable_instance(filter_value_colnames_vs_times=prediction_features, prediction_times=prediction_times)
        
        # Iterate through all curves, performing the internal SKLearn estimator's fit-predict on every single one.
        # Sklearn's requirement of (n_samples, n_features) corresponds to our (#splits, window_size * #time_features + #series_features).
        count_series = arr_X_fit.shape[0] # assume the # series is identical for all of the data arrays
        for series_idx in range(count_series):
            sub_arr_X_fit     = arr_X_fit[series_idx,:,:]
            sub_arr_Y_fit     = arr_Y_fit[series_idx,:,:]
        
            self.debug('Iterating over series: series_idx = ' + str(series_idx))
            self.debug('About to call fit on the inner ' + str(self._classic_estimator) + ' with the X & Y arrays having the shapes ' + str(sub_arr_X_fit.shape) + ' & ' + str(sub_arr_Y_fit.shape) + ' respectively.')
            self._classic_estimator.fit(sub_arr_X_fit, sub_arr_Y_fit)
            
            for prediction_times_subset in self.get_forward_sliding_windows(arr=prediction_times, max_window_size=self.width_output):
            
                self.debug('Iterating over output time windows: prediction_times_subset = ' + str(prediction_times_subset))
                
                # Combine this with predictions
                X_window = Y_hat.get_backward_time_window(self.width_input)
                
                # Convert (# series, # times, # features) to (#series, #times * #features) for the X time region only:
                arr_X_predict = self._selectTabularArrayForPredicting(X=X_window, include_time_as_feature=input_time_feature, value_colnames_filter=input_non_time_features)
            
                # Filter for the current series
                X_a2d = arr_X_predict[series_idx, :]
                
                # If necessary, reshape the 1-D array to a 2-D array representing a single sample
                if len(X_a2d.shape) == 1:
                    X_a2d = X_a2d.reshape(1, -1)
                
                # Do the multivariate prediction
                self.debug('About to call predict on the inner ' + str(self._classic_estimator) + ' with the X array having the shape ' + str(X_a2d.shape))
                Y_hat_a2d = self._classic_estimator.predict(X=X_a2d) 
                
                # Reshape to another 2D array form. 
                # From (#series == 1, #timestamps * # features) to (#timestamps, #features)
                Y_hat_a2d = Y_hat_a2d.reshape(self.width_output, len(prediction_features))
                
                # Take a subset of the predicted timestamps to maintain them within the prediction_times range
                Y_hat_a2d = Y_hat_a2d[np.arange(len(prediction_times_subset)), :]
                
                # Concatenate X_window + Y_hat_subset
                Y_hat.update_from_2d_array(series_idx=series_idx, a2d_vs_times=Y_hat_a2d, times=prediction_times_subset, value_colnames_vs_times=prediction_features)
                
            # END for prediction_times_subset    
        # END for series_idx
       
        # Filter by times, if necessary, in case we have predicted too far into the future
        (Y_hat_filtered, other)  = Y_hat.split_by_times(given_times=prediction_times)
        return Y_hat_filtered
    
    # Implementation of the abstract method.
    def get_deep_copy(self):
        res = SingleCurveTabularWindowedPredictor(classic_estimator=copy.deepcopy(self._classic_estimator), allow_missing_values=self._allow_missing_values)
        res.width_input = self.width_input
        res.width_output = self.width_output
        return res
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(width_input = ' + str(self.width_input) +', width_output = ' + str(self.width_output) + ', _classic_estimator = ' + str(self._classic_estimator) + ', allow_missing_values = ' + str(self._allow_missing_values) + ')')
  

        