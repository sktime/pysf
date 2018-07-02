
from .framework import AbstractTransformer

from scipy.interpolate import UnivariateSpline


class SmoothingSplineTransformer(AbstractTransformer):
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
    ParameterNameSplineDegree = 'spline_degree'
    ParameterNameSmoothingFactor = 'smoothing_factor'
    
    def __init__(self):
        super(AbstractTransformer, self).__init__()
        self.spline_degree = None
        self.smoothing_factor = None

    # Implementation of the abstract method
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            self.debug('Passed a None parameter_dict')
        else:
            # Copy the (mutable) dict so we can remove keys before passing the params through
            parameter_dict = dict(parameter_dict)
            if SmoothingSplineTransformer.ParameterNameSplineDegree in parameter_dict:
                self.spline_degree = int(parameter_dict[SmoothingSplineTransformer.ParameterNameSplineDegree])         # explicitly cast to int
                self.debug('Set self.spline_degree = ' + str(self.spline_degree))
                del parameter_dict[SmoothingSplineTransformer.ParameterNameSplineDegree]
            if SmoothingSplineTransformer.ParameterNameSmoothingFactor in parameter_dict:
                self.smoothing_factor = str(parameter_dict[SmoothingSplineTransformer.ParameterNameSmoothingFactor])   # explicitly cast to str
                self.debug('Set self.smoothing_factor = ' + str(self.smoothing_factor))
                del parameter_dict[SmoothingSplineTransformer.ParameterNameSmoothingFactor]
            # We do not try to pass up the parameter dict to any superclasses.
        
    def transform(self, X):          
        # Validate hyperparameters
        if self.spline_degree is None:
            raise Exception('spline_degree (int) cannot be set as None!')
        if self.smoothing_factor is None:
            raise Exception('smoothing_factor (str) cannot be set as None!')
        
        # Convert string smoothing_factor hyperparameter to an int:
        if self.smoothing_factor.lower() in ['default', 'none']:
            int_smoothing_factor = None
        else:
            int_smoothing_factor = int(self.smoothing_factor)
            
        (a3d_vs_times, a2d_vs_series, a1d_times) = X.select_arrays(include_time_as_feature=False, value_colnames_filter=None, allow_missing_values=True)
        self.debug('a3d_vs_times.shape = ' + str(a3d_vs_times.shape) + ', a1d_times.shape = ' + str(a1d_times))
        self.debug('k = ' + str(self.spline_degree) + ', s = ' + str(int_smoothing_factor))
        
        for series_idx in range(a3d_vs_times.shape[0]):
            for feature_idx in range(a3d_vs_times.shape[2]):
                y = a3d_vs_times[series_idx, :, feature_idx]
                y = UnivariateSpline(x=a1d_times, y=y, k=self.spline_degree, s=int_smoothing_factor)(a1d_times)
                #y = 123 + y
                a3d_vs_times[series_idx, :, feature_idx] = y
                #plt.plot(a1d_times, y)
    
        X_transformed = X.new_instance_from_3d_array(times=a1d_times, a3d_vs_times=a3d_vs_times, value_colnames_vs_times=None)
        return X_transformed
    
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(spline_degree = ' + str(self.spline_degree) +', smoothing_factor = ' + str(self.smoothing_factor) + ')')
  
    # Implementation of the abstract method.
    def get_deep_copy(self):
        res = SmoothingSplineTransformer()
        if self.spline_degree is not None:
            res.spline_degree = self.spline_degree
        if self.smoothing_factor is not None:
            res.smoothing_factor = self.smoothing_factor
        return res
    
        

            
################################
# For testing
################################
            
if False:
    
    from pysf.data import load_ramsay_weather_data_dfs, load_ramsay_growth_data_dfs, MultiSeries
    from sklearn.model_selection import KFold
    import pandas as pd

    ###################################
    # Data: weather
    ###################################
    
    (weather_vs_times_df, weather_vs_series_df) = load_ramsay_weather_data_dfs()
    data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')
    data_weather.visualise(title='Raw data')
    
    transformer = SmoothingSplineTransformer()
    
    params = {'spline_degree' : 5, 'smoothing_factor' : None}
    transformer.set_parameters(params)
    transformed_weather = transformer.transform(data_weather)
    transformed_weather.visualise(title=str(params))
    
    
    params = {'spline_degree' : 3, 'smoothing_factor' : 1200}
    transformer.set_parameters(params)
    transformed_weather = transformer.transform(data_weather)
    transformed_weather.visualise(title=str(params))
    
    
    ###################################
    # Subset of the weather data
    ###################################
    
    splits = list(data_weather.generate_series_folds(series_splitter=KFold(n_splits=5)))
    (other, subset_weather) = splits[0]
    subset_weather = subset_weather.get_forward_time_window(window_size=100)
    subset_weather.visualise(title='Subset of raw data')
    
    params = {'spline_degree' : 5, 'smoothing_factor' : None}
    transformer.set_parameters(params)
    transformed_weather = transformer.transform(subset_weather)
    transformed_weather.visualise(title=str(params))
    
    
    ###################################
    # Data: growth
    ###################################
    
    (growth_vs_times_df, growth_vs_series_df) = load_ramsay_growth_data_dfs()
    growth_vs_series_df['gender'] = growth_vs_series_df['gender'].astype('category')
    growth_vs_series_df = pd.concat([growth_vs_series_df, pd.get_dummies(growth_vs_series_df['gender'])], axis=1)
    data_growth = MultiSeries(data_vs_times_df=growth_vs_times_df, data_vs_series_df=growth_vs_series_df, time_colname='age', series_id_colnames=['gender', 'cohort_id'])
    data_growth.visualise(title='Raw data')
    
    params = {'spline_degree' : 3, 'smoothing_factor' : None}
    transformer.set_parameters(params)
    transformed_growth = transformer.transform(data_growth)
    transformed_growth.visualise(title=str(params))
    
    
    ###################################
    # Subset of the growth data
    ###################################
    
    splits = list(data_growth.generate_series_folds(series_splitter=KFold(n_splits=5)))
    (other, subset_growth) = splits[0]
    subset_growth = subset_growth.get_backward_time_window(window_size=10)
    subset_growth.visualise(title='Subset of raw data')
   
    params = {'spline_degree' : 3, 'smoothing_factor' : None}
    transformer.set_parameters(params)
    transformed_growth = transformer.transform(subset_growth)
    transformed_growth.visualise(title=str(params))
 
    
    
    
