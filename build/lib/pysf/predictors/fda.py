
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
            

    