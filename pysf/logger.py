
# Python's built in logging
import logging


# Module attributes
console_logging_enabled = True
console_log_level = logging.INFO
file_logging_enabled = True
file_log_level = logging.DEBUG
file_name = 'pysf.log'



# Create a base class
class LoggingHandler:
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
    def __init__(self, *args, **kwargs):
        # Set up logger with the default level of DEBUG, to let everything through.
        self.initLogger()
    
    def initLogger(self, loggerLevel=logging.DEBUG):
        # This gets the class name for the logger: 
        # https://stackoverflow.com/questions/7385037/how-do-i-get-the-name-of-the-class-containing-a-logging-call-in-python
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(loggerLevel)
        logger.handlers = []

        if (console_logging_enabled):
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            consoleHandler.setLevel(console_log_level)
            logger.addHandler(consoleHandler)
            
        if (file_logging_enabled):
            fileHandler = logging.FileHandler(file_name) # appends by default
            fileHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            fileHandler.setLevel(file_log_level)
            logger.addHandler(fileHandler)
        
        self.log = logger
        
    def debug(self, *args, **kwargs):
        self.log.debug(*args, **kwargs)
        
    def info(self, *args, **kwargs):
        self.log.info(*args, **kwargs)
        
    def warning(self, *args, **kwargs):
        self.log.warning(*args, **kwargs)
        
    def error(self, *args, **kwargs):
        self.log.error(*args, **kwargs)
        
        
# Does nothing special, we're just extending LoggingHandler so we have an 
# obvious class name ("GlobalLogger") to tie global logging statements to.
class GlobalLogger(LoggingHandler):
    def __init__(self):
        super(GlobalLogger, self).__init__()
    
        
        
# (Where in other languages we would use a singleton object for this,
# the Pythonic way is to use a module-bound variable)
global_logger = GlobalLogger()

        
      
##################################################
# For testing
##################################################

if False:
    
    # Create test class A that inherits the base class
    class testclassa(LoggingHandler):
        def testmethod1(self):
            # call self.log.<log level> instead of logging.log.<log level>
            self.debug("debug from test class A")
            self.info("info from test class A")
            self.warning("warning from test class A")
            self.error("error from test class A")
    
    
    # Create test class B that inherits the base class
    class testclassb(LoggingHandler):
        def __init__(self, *args, **kwargs):
            # Make this class more restrictive.
            self.initLogger(logging.WARNING)
            
        def testmethod2(self):
            # call self.log.<log level> instead of logging.log.<log level>
            self.debug("debug from test class B")
            self.info("info from test class B")
            self.warning("warning from test class B")
            self.error("error from test class B")
    
    
    testclassa().testmethod1()
    testclassb().testmethod2()
        