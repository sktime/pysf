
from .logger import global_logger

import numpy as np
import pickle
import os


# Type conversion, as some estimators do not like numpy types
def numpy_to_native(value):
    value_type = type(value)
    if value_type in [np.float32, np.float64, np.uint32]:
        return value.item()
    elif value_type in [np.int16]:
        return np.asscalar(value)
    else:
        return value
    

def create_parent_dir(filepath):
    parent_dirpath = os.path.dirname(filepath)
    if not os.path.exists(parent_dirpath):
        global_logger.info('Creating parent directory at ' + str(parent_dirpath))
        os.makedirs(parent_dirpath)

        
def to_pickle(filepath, obj, create_dir=True):
    try:
        global_logger.info('About to pickle-serialize ' + str(obj) + ' to ' + os.path.abspath(filepath))
        if create_dir:
            create_parent_dir(filepath)
        file = open(filepath, 'wb')
        pickle.dump(obj, open(filepath, 'wb'))
        file.close()
        global_logger.info('Done serializing' + str(obj))
    except Exception as ex:
        global_logger.error('Exception while trying to pickle-serialize: ' + str(ex))
        raise
        
    
def from_pickle(filepath):
    try:
        global_logger.info('About to pickle-deserialize from ' + os.path.abspath(filepath))
        file = open(filepath, 'rb')
        obj = pickle.load(file)
        file.close()
        global_logger.info('Done deserializing' + str(obj))
        return obj
    except Exception as ex:
        global_logger.error('Exception while trying to pickle-deserialize: ' + str(ex))
        raise
        
        
def get_friendly_list_string(seq):
    if seq is None:
        return str(seq)
    elif type(seq) == str:
        return seq
    elif type(seq) == list:
        return '+'.join(seq)
    else:
        raise Exception('Unexpected type ' + str(type(seq)))

        
def replace_string_curve_to_series(input_string):
    return input_string.replace('curve', 'series')
        
    
def limit_tf_mem():
    # http://forums.fast.ai/t/gpu-garbage-collection/1976/6
    from keras import backend as K
    
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def clear_tf_mem():
    # http://forums.fast.ai/t/gpu-garbage-collection/1976/6
    from keras import backend as K

    sess = K.get_session()
    sess.close()
    limit_tf_mem()
            

    
