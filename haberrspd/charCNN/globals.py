import os
from keras import backend as K

# The latest version of TF is crap, this silences the deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Check that at least one GPU is available
assert K.tensorflow_backend._get_available_gpus()
