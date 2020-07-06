# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint

# parameters
batch_size = 1024

# arguments
if len(sys.argv) != 5:
	print('Usg: python {} data model_json model_weights output_filename'.format(sys.argv[0]))
	exit()

# load data
data = np.load(sys.argv[1])

# load model
json_file = open(sys.argv[2], 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(sys.argv[3])

# predict
predictions = loaded_model.predict(data, batch_size=batch_size, verbose=1)
predictions = (predictions >= 0.5).astype('int')

# write
np.save(sys.argv[4], predictions)

