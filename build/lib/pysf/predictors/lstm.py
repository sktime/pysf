

from ..logger import LoggingHandler 
from .framework import MultiCurveTabularWindowedPredictor
from ..utils import clear_tf_mem

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Reshape
from keras.callbacks import Callback

from sklearn.preprocessing import MinMaxScaler

import math
import gc
from datetime import datetime



class TrainingLoggerCallback(LoggingHandler, Callback):
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
    def __init__(self, lstm_predictor):
        super(TrainingLoggerCallback, self).__init__()
        self._lstm_predictor_str = str(lstm_predictor)
        self._total_epochs = lstm_predictor.training_epochs
        
    def on_train_begin(self, logs={}):
        self._timer_start_training = datetime.now()
        self.debug('Training started')
        
    def on_epoch_begin(self, epoch, logs={}):
        self._timer_start_epoch = datetime.now()
        self.debug('Started Epoch ' + str(1+epoch) + '/' + str(self._total_epochs))
           
    def on_epoch_end(self, epoch, logs={}):
        self.debug('Finished Epoch ' + str(1+epoch) + '/' + str(self._total_epochs) + ': loss = ' + str(logs['loss']) + ', time taken = ' + str(datetime.now() - self._timer_start_epoch))
        
    def on_train_end(self, logs={}):
        self.debug('Training finished. Took ' + str(datetime.now() - self._timer_start_training))


class AbstractLstmPredictor(LoggingHandler):
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
    ParameterNameHiddenUnits = 'hidden_units'
    ParameterNameTrainingEpochs = 'training_epochs'
    ParameterNameInputDropout = 'input_dropout'
    ParameterNameRecurrentDropout = 'recurrent_dropout'
    
    def __init__(self):
        super(AbstractLstmPredictor, self).__init__()
        # Parameters
        self.hidden_units = None
        self.training_epochs = None
        self.input_dropout = None
        self.recurrent_dropout = None
        # Internal objects
        self._scalerOfInput = None
        self._scalerOfOutput = None
        self._classic_estimator = None    
        
    # Implementation of the abstract method. Simply passes the parameters through to the underlying sklearn estimator.
    def set_parameters(self, parameter_dict):          
        if parameter_dict is None:
            self.debug('Passed a None parameter_dict')
        else:
            # Copy the (mutable) dict so we can remove keys before passing the params through
            parameter_dict = dict(parameter_dict)
            if AbstractLstmPredictor.ParameterNameHiddenUnits in parameter_dict:
                # Explicitly cast to int or a list of ints:
                hu_val = parameter_dict[AbstractLstmPredictor.ParameterNameHiddenUnits]
                if type(hu_val) == list:
                    self.hidden_units = [ int(x) for x in hu_val ]
                else:
                    self.hidden_units = int(hu_val)
                self.debug('Set self.hidden_units = ' + str(self.hidden_units))
                del parameter_dict[AbstractLstmPredictor.ParameterNameHiddenUnits]
            if AbstractLstmPredictor.ParameterNameTrainingEpochs in parameter_dict:
                self.training_epochs = int(parameter_dict[AbstractLstmPredictor.ParameterNameTrainingEpochs])        # explicitly cast to int
                self.debug('Set self.training_epochs = ' + str(self.training_epochs))
                del parameter_dict[AbstractLstmPredictor.ParameterNameTrainingEpochs]
            if AbstractLstmPredictor.ParameterNameInputDropout in parameter_dict:
                self.input_dropout = float(parameter_dict[AbstractLstmPredictor.ParameterNameInputDropout])          # explicitly cast to float
                self.debug('Set self.input_dropout = ' + str(self.input_dropout))
                del parameter_dict[AbstractLstmPredictor.ParameterNameInputDropout]
            if AbstractLstmPredictor.ParameterNameRecurrentDropout in parameter_dict:
                self.recurrent_dropout = float(parameter_dict[AbstractLstmPredictor.ParameterNameRecurrentDropout])  # explicitly cast to float
                self.debug('Set self.recurrent_dropout = ' + str(self.recurrent_dropout))
                del parameter_dict[AbstractLstmPredictor.ParameterNameRecurrentDropout]
            # We do not try to pass up the parameter dict to any superclasses.
        
    def _buildAndFitLstmEstimator(self, a3d_vs_times_input, a3d_vs_times_output):
        # Validate hyperparameters, part 1
        if None in [self.hidden_units, self.training_epochs, self.input_dropout, self.recurrent_dropout]:
            raise Exception('All of the following parameters must be set! hidden_units = '+ str(self.hidden_units) + ', training_epochs = ' + str(self.training_epochs) + ', input_dropout = ' + str(self.input_dropout) + ', recurrent_dropout = ' + str(self.recurrent_dropout))
        
        # Validate hyperparameters, part 2
        if type(self.hidden_units) != list:
            self.hidden_units = [ int(self.hidden_units) ] # explicitly cast a single-element value to int, and wrap in a list
            self.debug('After validation, self.hidden_units = ' + str(self.hidden_units))
        self.debug('Will build a ' + str(len(self.hidden_units)) + '-layer LSTM')
            
        # Validate hyperparameters, part 3
        # Otherwise, we will get an error when adding the Reshape layer to our model architecture: "ValueError: total size of new array must be unchanged"
        if self.width_output > self.width_input:
            raise Exception('width_output=' + str(self.width_output) + ' is greather than width_input=' + str(self.width_input))
        if math.modf(self.width_input / self.width_output)[0] != 0.0:
            raise Exception('width_input=' + str(self.width_input) + ' is not an exact multiple of width_output=' + str(self.width_output))
            
        # Get info about the training inputs, which should have shape (# samples, # timestamps, # features)
        shape_a3d_vs_times_input  = a3d_vs_times_input.shape
        shape_a3d_vs_times_output = a3d_vs_times_output.shape
        count_input_features  = shape_a3d_vs_times_input[2]
        count_output_features = shape_a3d_vs_times_output[2]
        self.debug('shape_a3d_vs_times_input = ' + str(shape_a3d_vs_times_input) + ', count_input_features = ' + str(count_input_features) + ', shape_a3d_vs_times_output = ' + str(shape_a3d_vs_times_output) + ', count_output_features = ' + str(count_output_features))

        # Normalise the data in the range 0..1. To do this, we need to temporarily reshape it to 2-D.
        self._scalerOfInput  = MinMaxScaler(feature_range=(0, 1))
        self._scalerOfOutput = MinMaxScaler(feature_range=(0, 1))
        a2d_vs_times_input  = a3d_vs_times_input.reshape(shape_a3d_vs_times_input[0], shape_a3d_vs_times_input[1] * shape_a3d_vs_times_input[2])
        a2d_vs_times_output = a3d_vs_times_output.reshape(shape_a3d_vs_times_output[0], shape_a3d_vs_times_output[1] * shape_a3d_vs_times_output[2])
        scaled_a2d_vs_times_input  = self._scalerOfInput.fit_transform(a2d_vs_times_input)
        scaled_a2d_vs_times_output = self._scalerOfOutput.fit_transform(a2d_vs_times_output)
        scaled_a3d_vs_times_input  = scaled_a2d_vs_times_input.reshape(shape_a3d_vs_times_input)
        scaled_a3d_vs_times_output = scaled_a2d_vs_times_output.reshape(shape_a3d_vs_times_output)

        # Build the LSTM model
        # This is a good discussion reference on a multivariate LSTM, such as we have here: https://github.com/fchollet/keras/issues/2892
        model = Sequential()
        try:
            
            for layer_idx in range(len(self.hidden_units)):
                layer_hidden_units = self.hidden_units[layer_idx]
                if layer_idx == 0:
                    model.add(LSTM(input_shape=(self.width_input, count_input_features), units=layer_hidden_units, dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True))
                else:
                    model.add(LSTM(units=layer_hidden_units, dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True))
                    
            model.add(Reshape((self.width_output, -1)))
            model.add(TimeDistributed(Dense(count_output_features)))
        except:
            self.error('Model architecture so far is: ')
            try:
                model.summary(print_fn=self.error)
            except:
                self.error('Error while trying to display a summary of the offending model')
            raise # re-raise the original exception
             
        self.debug('Built the following model:')
        model.summary(print_fn=self.debug)
        
        model.compile(loss='mse', optimizer='adam')
        self._classic_estimator = model

        # Fit the model to the normalised data       
        logging_callback = TrainingLoggerCallback(lstm_predictor=self)
        self._classic_estimator.fit(x=scaled_a3d_vs_times_input, y=scaled_a3d_vs_times_output, epochs=self.training_epochs, callbacks=[logging_callback])
        self.debug('Done building and fitting an LSTM model, and calibrating the preprocessors')
        
    def _predictFromLstmEstimator(self, a3d_vs_times_input):
        # Get info about the test input, which should have shape (# samples, # timestamps, # features)
        shape_a3d_vs_times_input  = a3d_vs_times_input.shape
        self.debug('shape_a3d_vs_times_input = ' + str(shape_a3d_vs_times_input))
        
        # Normalise the data in the range 0..1. To do this, we need to temporarily reshape it to 2-D.
        a2d_vs_times_input  = a3d_vs_times_input.reshape(shape_a3d_vs_times_input[0], shape_a3d_vs_times_input[1] * shape_a3d_vs_times_input[2])
        scaled_a2d_vs_times_input  = self._scalerOfInput.fit_transform(a2d_vs_times_input)
        scaled_a3d_vs_times_input  = scaled_a2d_vs_times_input.reshape(shape_a3d_vs_times_input)
        
        # Perform the prediction. This requires normalised inputs and returns a normalised output
        scaled_a3d_vs_times_output = self._classic_estimator.predict(x=scaled_a3d_vs_times_input)
        shape_a3d_vs_times_output = scaled_a3d_vs_times_output.shape
        self.debug('shape_a3d_vs_times_output = ' + str(shape_a3d_vs_times_output))
        
        # Invert the normalisation of the output test data, to return it to its original scale. This involves a temporary reshape to 2-D.
        scaled_a2d_vs_times_output = scaled_a3d_vs_times_output.reshape(shape_a3d_vs_times_output[0], shape_a3d_vs_times_output[1] * shape_a3d_vs_times_output[2])
        a2d_vs_times_output = self._scalerOfOutput.inverse_transform(scaled_a2d_vs_times_output)
        a3d_vs_times_output = a2d_vs_times_output.reshape(shape_a3d_vs_times_output)
        
        return a3d_vs_times_output
        
    
        

class MultiCurveWindowedLstmPredictor(MultiCurveTabularWindowedPredictor, AbstractLstmPredictor):
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
        MultiCurveTabularWindowedPredictor.__init__(self, classic_estimator=None, allow_missing_values=allow_missing_values)
        AbstractLstmPredictor.__init__(self)
        
    def set_parameters(self, parameter_dict): 
        # Explicitly call both super methods:
        MultiCurveTabularWindowedPredictor.set_parameters(self, parameter_dict=parameter_dict)
        AbstractLstmPredictor.set_parameters(self, parameter_dict=parameter_dict)
        
    def _selectTabularWindowedArraysForFitting(self, X, include_time_as_feature, value_colnames_filter):
        #(X_arr, Y_arr) = X.select_paired_tabular_windowed_2d_arrays(include_time_as_feature=include_time_as_feature,  value_colnames_filter=value_colnames_filter,  allow_missing_values=self._allow_missing_values,  input_sliding_window_size=self.width_input, output_sliding_window_size=self.width_output)
        (X_arr, Y_arr) = X.select_paired_tabular_windowed_3d_by_time_arrays(include_time_as_feature=include_time_as_feature,  value_colnames_filter=value_colnames_filter,  allow_missing_values=self._allow_missing_values,  input_sliding_window_size=self.width_input, output_sliding_window_size=self.width_output)
        return (X_arr, Y_arr)
    
    def _fitClassicEstimator(self, X_arr, Y_arr):
        self._buildAndFitLstmEstimator(a3d_vs_times_input=X_arr, a3d_vs_times_output=Y_arr)
        
    def _selectTabularArrayForPredicting(self, X, include_time_as_feature, value_colnames_filter):
        #(X_a2d, t) = X.select_tabular_full_2d_array(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter) 
        (X_a2d, t) = X.select_merged_3d_array(include_time_as_feature=include_time_as_feature, value_colnames_filter=value_colnames_filter) 
        return X_a2d
 
    def _predictFromClassicEstimator(self, X_arr):
        a3d_vs_times_output = self._predictFromLstmEstimator(a3d_vs_times_input=X_arr)
        return a3d_vs_times_output
        
    # Implementation of the abstract method.
    def get_deep_copy(self):
        res = MultiCurveWindowedLstmPredictor(allow_missing_values=self._allow_missing_values)
        # (There's no need to copy over a copy of the classic estimator since only 1 instance lives per "fit")
        # Specific to this predictor:
        res.hidden_units = self.hidden_units
        res.training_epochs = self.training_epochs
        res.input_dropout = self.input_dropout
        res.recurrent_dropout = self.recurrent_dropout
        # From MultiCurveTabularWindowedPredictor:
        res.width_input = self.width_input
        res.width_output = self.width_output
        res.train_over_prediction_times = self.train_over_prediction_times
        return res        
        
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(hidden_units = ' + str(self.hidden_units) +', training_epochs = ' + str(self.training_epochs) + ', input_dropout = ' + str(self.input_dropout) + ', recurrent_dropout = ' + str(self.recurrent_dropout) + ', width_input = ' + str(self.width_input) + ', width_output = ' + str(self.width_output)  + ', train_over_prediction_times = ' + str(self.train_over_prediction_times) +  ', classic_estimator = ' + str(self._classic_estimator) + ', allow_missing_values = ' + str(self._allow_missing_values) + ')')
  
    # Override of non-abstract method
    def compact(self):
        self.debug('Started compacting')
        del self._classic_estimator
        self._classic_estimator = None
        try:
            clear_tf_mem()
        except:
            self.warn('Could not clear TensorFlow memory')
        self.debug('GC output: ' + str(gc.collect()))
        self.debug('Done compacting')
            
            
 
    