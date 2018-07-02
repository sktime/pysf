

from ..logger import LoggingHandler 

from abc import ABC, abstractmethod


class AbstractTransformer(ABC, LoggingHandler):
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
        super(AbstractTransformer, self).__init__()

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
    
    @abstractmethod
    def transform(self, X):  
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

