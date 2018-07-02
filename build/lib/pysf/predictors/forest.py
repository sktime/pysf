
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
    

