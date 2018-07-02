

from .logger import LoggingHandler 

from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
import numpy as np
import unittest

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# This wraps around a sklearn.model_selection.KFold splitter object and is used
# in exactly the same way. The only difference is that it swaps around 
# training and test indices, putting the "reverse" in the name.
class ReverseKFold:
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
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        self._kFold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return self._kFold.get_n_splits(X=X, y=y, groups=groups)
        
    def split(self, X, y=None, groups=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Yields:
            True if successful, False otherwise.

        """
        for train, test in self._kFold.split(X=X, y=y, groups=groups):
            # Instead of yield train, test
            yield test, train


class AbstractTimeSeriesSplit(ABC, LoggingHandler):
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
    def __init__(self, count_timestamps, training_set_size, validation_set_size, step):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        super(AbstractTimeSeriesSplit, self).__init__()

        # Validation
        if count_timestamps < 1:
            raise Exception('Should have at least 1 timestamp! Given ' + str(count_timestamps) + ' instead.')
        if training_set_size < 1:
            raise Exception('Training set size should be at least 1! Given ' + str(training_set_size) + ' instead.')
        if validation_set_size < 1:
            raise Exception('Test set size should be at least 1! Given ' + str(validation_set_size) + ' instead.')
        if (training_set_size + validation_set_size) > count_timestamps:
            raise Exception('The ' + str(count_timestamps) + ' observations are insufficient for a rolling window cross-valiation with test set size (forecast horizon) ' + str(validation_set_size) + ' and training set size (rolling window length) ' + str(training_set_size) + '.')

        # Using single-underscore instead of double- to make these easier to access from subclasses
        self._count_timestamps = count_timestamps
        self._validation_set_size = validation_set_size
        self._training_set_size = training_set_size
        self._step = step
        self._num_splits = self.__num_splits() # caching for efficiency

        
    # This syntax is generator-related. Allows the user to iterate over an instance obj of our class.
    # Also, it's an abstract method
    @abstractmethod
    def __iter__(self):            
        pass

    # Returns the number of splits
    def __num_splits(self):
        num_splits = int((self._count_timestamps - (self._training_set_size + self._validation_set_size)) / self._step) + 1 # int() truncates the float, as we want.
        return num_splits
    
    # Also returns the number of splits, as a convenience method
    # This syntax allows len(obj) to be called on an instance obj of our class
    def __len__(self):
        return self._num_splits
    
    # This syntax allows str(obj) to be called on an instance obj of our class
    def __repr__(self):
        return (self.__class__.__name__ + '(count_timestamps=' + str(self._count_timestamps) + ', validation_set_size=' + str(self._validation_set_size) + ', training_set_size=' + str(self._training_set_size) + ', step=' + str(self._step) + ' => num_splits=' + str(self._num_splits) + ')')
    
    def visualise(self, ax=None):
        """Class methods are similar to regular functions.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        # Collect the generator's output into a list of tupes:
        split_indices_list = [(train, test) for train, test in self]
    
        viz_X = np.zeros((self._count_timestamps, len(split_indices_list)))
        #print(viz_X)
    
        idx_col = 0
        for (train, test) in split_indices_list:
            # Play around with these colours in conjunction with the colour map!
            viz_X[:, idx_col] =  1 # neither/default
            viz_X[train, idx_col] =  0
            viz_X[test, idx_col] =  2
            idx_col = idx_col + 1
    
        #print(viz_X)
        #plt.matshow(np.transpose(viz_X), cmap=cm.Spectral_r)
        
        # Plot on an axis object if given one
        (ax, plt)[ax is None].matshow(np.transpose(viz_X), cmap=cm.bwr, aspect='auto')   # cmap=cm.Spectral_r
        
        # Totally different APIs depending on whether these are subplots or not
        ylab = 'Iteration'
        xlab = 'Data (row) index'
        #title = str(self._training_set_size) + '/' + str(self._validation_set_size) + ' windows over ' + str(self._n) + ' rows, step size ' + str(self._step)
        title = str(self)
        if (ax is None):
            plt.ylabel(ylab)
            plt.xlabel(xlab)
            plt.title(title)
            plt.gca().xaxis.tick_bottom() # show the x-axis along the bottom instead of the top
        else:
            ax.set_ylabel(ylab)
            ax.set_xlabel(xlab)
            ax.set_title(title)
            ax.axes.get_xaxis().tick_bottom()

        
        
# Concrete implementation
class SlidingWindowTimeSeriesSplit(AbstractTimeSeriesSplit):  
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
    def __init__(self, count_timestamps, training_set_size=1, validation_set_size=1, step=1):
        super(SlidingWindowTimeSeriesSplit, self).__init__(count_timestamps=count_timestamps, training_set_size=training_set_size, validation_set_size=validation_set_size, step=step)
        
    # Implementation of the abstract method
    def __iter__(self):
        # range() is also implemented as a generator. It doesn't take keyword args, but they are "start", "stop" and "step", resp.
        for split in range(0, self._num_splits, 1):
            idx_train_start_inc = split * self._step
            idx_train_end_inc  = idx_train_start_inc + self._training_set_size - 1
            idx_test_start_inc = idx_train_start_inc + self._training_set_size
            idx_test_end_inc   = idx_test_start_inc  + self._validation_set_size  - 1
            #yield ( (idx_train_start_inc, idx_train_end_inc), (idx_test_start_inc, idx_test_end_inc) )
        
            idx_train_all = np.arange(idx_train_start_inc, idx_train_end_inc + 1)
            idx_test_all =  np.arange(idx_test_start_inc,  idx_test_end_inc  + 1)
            yield (idx_train_all, idx_test_all)
    

# Concrete implementation
class ExpandingWindowTimeSeriesSplit(AbstractTimeSeriesSplit):  
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
    def __init__(self, count_timestamps, training_set_size=1, validation_set_size=1, step=1):
        super(ExpandingWindowTimeSeriesSplit, self).__init__(count_timestamps=count_timestamps, training_set_size=training_set_size, validation_set_size=validation_set_size, step=step)
        
    # Implementation of the abstract method
    def __iter__(self):
        # range() is also implemented as a generator. It doesn't take keyword args, but they are "start", "stop" and "step", resp.
        for split in range(0, self._num_splits, 1):
            # Same window calculation logic as a sliding window...
            idx_train_start_inc = split * self._step
            idx_train_end_inc  = idx_train_start_inc + self._training_set_size - 1
            idx_test_start_inc = idx_train_start_inc + self._training_set_size
            idx_test_end_inc   = idx_test_start_inc  + self._validation_set_size  - 1
            
            # ... until we reset the training start index to 0 (hence the expanding window):
            idx_train_start_inc = 0
        
            idx_train_all = np.arange(idx_train_start_inc, idx_train_end_inc + 1)
            idx_test_all =  np.arange(idx_test_start_inc,  idx_test_end_inc  + 1)
            yield (idx_train_all, idx_test_all)
    

##################################################
# For testing
##################################################
  
class TestSlidingWindowTimeSeriesSplit(unittest.TestCase):
    def test_splits_18_observations_step_1(self):
        print('Running ' + self.test_splits_18_observations_step_1.__name__)
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=18)
        self.assertEqual(17, len(swin))
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=18, training_set_size=3, validation_set_size=2)
        self.assertEqual(14, len(swin))
    def test_splits_7_observations_step_1(self):
        print('Running ' + self.test_splits_7_observations_step_1.__name__)
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=7)
        self.assertEqual(6, len(swin))
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=7, training_set_size=2, validation_set_size=1)
        self.assertEqual(5, len(swin))
    def test_splits_18_observations_step_2(self):
        print('Running ' + self.test_splits_18_observations_step_2.__name__)
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=18, step=2)
        self.assertEqual(9, len(swin))
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=18, training_set_size=3, validation_set_size=2, step=2)
        self.assertEqual(7, len(swin))
    def test_splits_7_observations_step_2(self):
        print('Running ' + self.test_splits_7_observations_step_2.__name__)
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=7, step=2)
        self.assertEqual(3, len(swin))
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=7, training_set_size=2, validation_set_size=1, step=2)
        self.assertEqual(3, len(swin))
    def test_splits_18_observations_step_5(self):
        print('Running ' + self.test_splits_18_observations_step_5.__name__)
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=18, step=5)
        self.assertEqual(4, len(swin))
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=18, training_set_size=3, validation_set_size=2, step=5)
        self.assertEqual(3, len(swin))
    def test_splits_7_observations_step_5(self):
        print('Running ' + self.test_splits_7_observations_step_5.__name__)
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=7, step=5)
        self.assertEqual(2, len(swin))
        swin = SlidingWindowTimeSeriesSplit(count_timestamps=7, training_set_size=2, validation_set_size=1, step=5)
        self.assertEqual(1, len(swin))
    def runTest(self):
        self.test_splits_18_observations_step_1()   
        self.test_splits_7_observations_step_1() 
        self.test_splits_18_observations_step_2()   
        self.test_splits_7_observations_step_2() 
        self.test_splits_18_observations_step_5()   
        self.test_splits_7_observations_step_5() 

        
        
if __name__ == "__main__":

    if True:
        TestSlidingWindowTimeSeriesSplit().run()
    else:
        TestSlidingWindowTimeSeriesSplit().test_splits_18_observations_step_1()   
        TestSlidingWindowTimeSeriesSplit().test_splits_7_observations_step_1() 
        TestSlidingWindowTimeSeriesSplit().test_splits_18_observations_step_2()   
        TestSlidingWindowTimeSeriesSplit().test_splits_7_observations_step_2() 
        TestSlidingWindowTimeSeriesSplit().test_splits_18_observations_step_5()   
        TestSlidingWindowTimeSeriesSplit().test_splits_7_observations_step_5() 
    
          
    if True:
            
        for step in [1, 2, 3]:
            for (count_timestamps, training_set_size, validation_set_size) in [(5, 1, 1), (18, 3, 2), (50, 10, 5)]:
                # Rolling window
                swin = SlidingWindowTimeSeriesSplit(count_timestamps=count_timestamps, training_set_size=training_set_size, validation_set_size=validation_set_size, step=step)
                print(swin)
                swin.visualise()
                for (idx_train_all, idx_test_all) in swin:
                    print('Train = ' + str(idx_train_all) + ', Test = ' + str(idx_test_all))
                print()
                
                # Expanding window
                ewin = ExpandingWindowTimeSeriesSplit(count_timestamps=count_timestamps, training_set_size=training_set_size, validation_set_size=validation_set_size, step=step)
                print(ewin)
                ewin.visualise()
                for (idx_train_all, idx_test_all) in ewin:
                    print('Train = ' + str(idx_train_all) + ', Test = ' + str(idx_test_all))
                print()
    
