
from .framework import SingleCurveSeriesPredictor, MultiCurveTabularPredictor

import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.dummy import DummyRegressor


####################################################################################################
# BASELINES IN GRID SETTING:
#   (1) Single-series: Linear interpolation. First/last if at edges.
#   (2) Single-series: Predict mean of series.
#   (3) Multi-series: predict mean of that time
#   (4) Predict 0.
####################################################################################################


class LinearInterpolator(BaseEstimator):
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
    
    @staticmethod
    def convert_to_1d_array(X):
        if X.ndim == 1:
            return X # we've verified it's a 1-D array
        elif X.ndim == 2 and X.shape[1] == 1:
            return X.ravel() # convert from a 2-D array shaped (n,1) to a 1-D array shaped (n,)
        else:
            raise Exception('X is ' + str(X.ndim) + '-D but we require 1-D for single-dimensional interpolation!')
        
    # X is required to be a 1-D array sorted in ascending order
    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        X = LinearInterpolator.convert_to_1d_array(X)        
        
        # Cache the X & y arrays for use later
        self._fitted_X = X
        self._fitted_y = y
        
        # Extract values of interest
        self._fitted_X_at_start = self._fitted_X[0]
        self._fitted_X_at_end = self._fitted_X[-1]
        self._fitted_y_at_start = self._fitted_y[0]
        self._fitted_y_at_end = self._fitted_y[-1]
        
        # Initialise a linear interpolator
        self._interpolator = interpolate.interp1d(x=self._fitted_X, y=self._fitted_y, axis=0, kind='linear')
        
        # Return the estimator
        return self

    def predict(self, X):
        X = LinearInterpolator.convert_to_1d_array(X)
        
        # Get flags to index the array values so we can treat them differently
        is_earlier = X < self._fitted_X_at_start
        is_later = X > self._fitted_X_at_end
        is_within = np.logical_not(np.logical_or(is_earlier, is_later))
        
        # Declare y with the right shape
        y = np.empty((X.shape[0], self._fitted_y.shape[1]), self._fitted_y.dtype)
        y.fill(np.nan)
        
        # Copy over values at the boundaries from the fitted values,
        # and calculate values within by interpolating the fitted values
        y[is_earlier] = self._fitted_y_at_start
        y[is_later] = self._fitted_y_at_end
        y[is_within] = self._interpolator(x=X[is_within])
        
        return y
        
    # Convenience method that carries out the prediction, and then outputs some charts
    def predict_and_visualise(self, X):
        # Carry out the prediction
        y = self.predict(X)
        
        # Do the visualisation
        fig=plt.figure()
        count_dims = self._fitted_y.shape[1]
        for dim_idx in range(count_dims):
            ax = fig.add_subplot(count_dims, 1, dim_idx+1) 
            sub_y_fitted = self._fitted_y[:, dim_idx] 
            sub_y_new = y[:, dim_idx] 
            ax.plot(self._fitted_X, sub_y_fitted, c='blue')
            ax.scatter(x=self._fitted_X, y=sub_y_fitted, c='blue')
            ax.scatter(x=X, y=sub_y_new, c='red')
            if dim_idx == 0:
                ax.set_title('Result of performing multiple single-series linear interpolations')
        plt.tight_layout()
        
        
class SeriesLinearInterpolator(SingleCurveSeriesPredictor):
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
        super(SeriesLinearInterpolator, self).__init__(classic_estimator = LinearInterpolator(), allow_missing_values = allow_missing_values)

    def get_deep_copy(self):
        res = SeriesLinearInterpolator(allow_missing_values=self._allow_missing_values)
        res._classic_estimator = copy.copy(self._classic_estimator)
        return res
        
        
class ZeroPredictor(SingleCurveSeriesPredictor):
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
        # Initialise with no estimator at the moment as we do not have the information necessary to initialise it yet!
        super(ZeroPredictor, self).__init__(classic_estimator=None, allow_missing_values = allow_missing_values)
    
    # Override the fit implementation as we actually need to fit the output dimensions of the DummyRegressor here
    def _fitImplementation(self, X, prediction_times, input_time_feature=True, input_non_time_features=None, prediction_features=None):
        if prediction_features is None:
            count_prediction_features = X.count_features
        elif type(prediction_features) == str:
            count_prediction_features = 1
        else:
            count_prediction_features = len(prediction_features)
        self._classic_estimator = DummyRegressor(strategy='constant', constant=np.zeros(count_prediction_features))
        self.debug('Counted '+  str(count_prediction_features) + ' features so have initialised an ' + str(self._classic_estimator))
    
    def get_deep_copy(self):
        res = ZeroPredictor(allow_missing_values=self._allow_missing_values)
        res._classic_estimator = copy.copy(self._classic_estimator)
        return res
        
        
class SeriesMeansPredictor(SingleCurveSeriesPredictor):
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
        super(SeriesMeansPredictor, self).__init__(classic_estimator = DummyRegressor(strategy='mean'), allow_missing_values = allow_missing_values)

    def get_deep_copy(self):
        res = SeriesMeansPredictor(allow_missing_values=self._allow_missing_values)
        res._classic_estimator = copy.copy(self._classic_estimator)
        return res

        
class TimestampMeansPredictor(MultiCurveTabularPredictor):
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
        super(TimestampMeansPredictor, self).__init__(classic_estimator = DummyRegressor(strategy='mean'), allow_missing_values = allow_missing_values)

    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):    
        # Call the parent class method
        Y_hat = super(TimestampMeansPredictor, self)._predictImplementation(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        # Filter out all the NaN other times that the DummyRegressor outputs:
        (Y_hat_filtered, other) = Y_hat.split_by_times(given_times=prediction_times)
        return Y_hat_filtered
        
        
    def get_deep_copy(self):
        # No need to explicitly copy over the DummyRegressor here
        res =  TimestampMeansPredictor(allow_missing_values=self._allow_missing_values)
        res._classic_estimator = copy.copy(self._classic_estimator)
        return res

    
