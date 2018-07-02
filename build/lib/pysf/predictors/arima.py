
from .framework import SingleCurveSeriesPredictor

import numpy as np
import statsmodels.api as sm
from sklearn.dummy import DummyRegressor


class ArimaPredictor(SingleCurveSeriesPredictor):
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
    ParameterNameOrderP = 'p'
    ParameterNameOrderD = 'd'
    ParameterNameOrderQ = 'q'

    def __init__(self, allow_missing_values=False):
        super(ArimaPredictor, self).__init__(classic_estimator=None)
        self._allow_missing_values = allow_missing_values
        
        # Need these set to something
        self.p = None
        self.d = None
        self.q = None
        
        
    # Override the implementation of this abstract method
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            raise Exception('Passed a None parameter_dict')   
        self.p = int(parameter_dict[ArimaPredictor.ParameterNameOrderP])
        self.d = int(parameter_dict[ArimaPredictor.ParameterNameOrderD])
        self.q = int(parameter_dict[ArimaPredictor.ParameterNameOrderQ])
        self.debug('Have set values from the parameter dictionary ' + str(parameter_dict)) 
        
        
    # Override of this method to call fit() on the classic estimator.
    def _innerClassicFit(self, a2d_X_fit, a2d_Y_fit, prediction_features):
        # Initialise and fit the ARIMA model here
        self.debug('About to initialise and call fit on an ARIMA model')
        struc_X_fit = np.core.records.fromarrays(a2d_X_fit.transpose(), names=(','.join(prediction_features)))
        arima = sm.tsa.ARIMA(struc_X_fit, order=(self.p, self.d, self.q))
        # Fitting by CSS since the MLE fitting types sometimes do not converge:
        fitting_method = 'css' 
        try:
            self._classic_estimator = arima.fit(disp=False, method=fitting_method, transparams=False)
        except:
            # Deal with the following exception (which is annoyingly not a warning): ValueError: The computed initial AR coefficients are not stationary You should induce stationarity, choose a different model order, or you can pass your own start_params.
            # Set starting params to 0 when the initial AR params are non-stationary.
            override_start_params = np.zeros(self.p + self.d + self.q)
            self.warning('Issue when fitting the ARIMA model. Will now override the starting params to ' + str(override_start_params))
            try:
                self._classic_estimator = arima.fit(disp=False, method=fitting_method, transparams=False, start_params=override_start_params)
            except:
                self.warning('Issue when fitting the ARIMA model, even with overriden starting parameters. Will rely on the baseline')
                self._classic_estimator = None
                
        # Also fit a simple model of series means. This will allow us to fall back to it if the ARIMA model is unstable
        self._fallback_estimator = DummyRegressor(strategy='mean')
        self._fallback_estimator.fit(X=a2d_X_fit, y=a2d_Y_fit)
                
                    
    # Override of this method to call predict() on the classic estimator.
    def _innerClassicPredict(self, a2d_X_predict, prediction_time_start_idx, prediction_time_end_idx):
        self.debug('About to call predict on the inner ' + str(self._classic_estimator) + ' with assumed time index boundaries ' + str((prediction_time_start_idx, prediction_time_end_idx)))
        # statsmodels requires these to be Python int types, not numpy ints:
        prediction_time_start_idx = int(prediction_time_start_idx)
        prediction_time_end_idx = int(prediction_time_end_idx)
        
        # We may have a None ARIMA model at this point, if there were problems with fitting
        use_fallback = (self._classic_estimator is None)
        if not(use_fallback):
            try:
                # statsmodels considers d==0 to imply an ARMA model's API, not the ARIMA model we use! hence the "typ" param disappearing:
                #   - typ: predict levels of the actual data and not differences
                #   - dynamic: just in case we're predicting in-sample, use actual forecasts
                if self.d == 0:
                    current_Y_hat = self._classic_estimator.predict(dynamic=True, start=prediction_time_start_idx, end=prediction_time_end_idx) 
                else:
                    current_Y_hat = self._classic_estimator.predict(typ='levels', dynamic=True, start=prediction_time_start_idx, end=prediction_time_end_idx) 
            except:
                use_fallback = True
        
        # If there has been some sort of error (either at fit or predict) so far, use the baseline
        if use_fallback:
            self.warning('Issue when predicting from the ARIMA model. Will now predict the fallback of series means.')
            current_Y_hat = self._fallback_estimator.predict(X=a2d_X_predict)
            
        self.debug('Return value from calling predict on the inner ' + str(self._classic_estimator) + ' has the shape ' + str(current_Y_hat.shape))
        return current_Y_hat
        
        
    # Implementation of the abstract method
    def _predictImplementation(self, X, prediction_times, input_time_feature, input_non_time_features, prediction_features):
        # ARIMA-specific validation: no time feature is allowed, and prediction and input (non-time) features must be identical
        if input_time_feature:
            raise Exception('We cannot include time as an input feature to an ARIMA model')
        if input_non_time_features is None or len(input_non_time_features) == 0:
            raise Exception('Must specify input_non_time_features. It was given as ' + str(input_non_time_features))
        if prediction_features is None or len(prediction_features) == 0:
            raise Exception('Must specify prediction_features. It was given as ' + str(prediction_features))
        set_input_non_time_features = set(input_non_time_features)
        set_prediction_features = set(prediction_features)
        diff = set_prediction_features.symmetric_difference(set_input_non_time_features)
        if len(diff) > 0:
            raise Exception('Differences were found between prediction_features and input_non_time_features: ' + str(diff))
        if self.p is None or self.d is None or self.q is None:
            raise Exception('The ARIMA order parameters have not yet been set: (p, d, q) = ' + str((self.p, self.d, self.q)))
        if len(prediction_features) != 1:
            raise Exception('The ARIMA prediction model only supports a single endogenous variable. Instead ' + str(len(prediction_features)) + ' features were specified.')
            
        # Call the parent class method
        Y_hat = super(ArimaPredictor, self)._predictImplementation(X=X, prediction_times=prediction_times, input_time_feature=input_time_feature, input_non_time_features=input_non_time_features, prediction_features=prediction_features)
        return Y_hat
        
        
    # Implementation of the abstract method.
    def get_deep_copy(self):
        res = ArimaPredictor(allow_missing_values=self._allow_missing_values)
        # (There's no need to copy over a copy of the classic estimator since only 1 instance lives per "fit")
        res.p = self.p
        res.d = self.d
        res.q = self.q
        return res
        
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(p = ' + str(self.p) +', d = ' + str(self.d) +', q = ' + str(self.q)  + ', classic_estimator = ' + str(self._classic_estimator) + ', allow_missing_values = ' + str(self._allow_missing_values) + ')')
  
        
        