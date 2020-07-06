# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import numpy as np
import keras.backend as K
from keras.layers import Input, LSTM, Dense, Permute, multiply, Flatten
from keras.layers import Reshape, GlobalAveragePooling1D
from keras.layers import Conv1D, BatchNormalization, Activation, add
from keras.optimizers import Adam
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# parameters for grid search
'''
batch_size = [128, 256, 512]
lr = [1e-3, 1e-4]
hidden_units = [64, 128, 256]
epochs = [100]
'''
batch_size = [128]
lr = [1e-3]
hidden_units = [128]
epochs = [10]

# functions
# https://sike6054.github.io/blog/paper/seventh-post/
# resnet 50 se

def conv1d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
	x = Conv1D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides)(x)
	x = BatchNormalization()(x)
	if activation:
		x = Activation(activation)(x)

	return x

def se_block(input_tensor, reduction_ratio=16):
	ch_input = K.int_shape(input_tensor)[-1]
	ch_reduced = ch_input // reduction_ratio

	# squeeze
	x = GlobalAveragePooling1D()(input_tensor)

	# excitation
	x = Dense(ch_reduced, kernel_initializer='he_normal', activation='relu', use_bias=False)(x)
	x = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(x)

	x = multiply([input_tensor, x])

	return x

def se_residual_block(input_tensor, hidden_units, strides=1, reduction_ratio=16):
	x = conv1d_bn(input_tensor, hidden_units, kernel_size=8)
	x = conv1d_bn(x, hidden_units, kernel_size=5)
	x = conv1d_bn(x, hidden_units, kernel_size=3, activation=None)

	x = se_block(x, reduction_ratio)

	projected_input = conv1d_bn(input_tensor, hidden_units, kernel_size=1, activation=None)
	shortcut = add([projected_input, x])
	shortcut = Activation(activation='relu')(shortcut)

	return shortcut

def build_se_resnet(seq_shape, hidden_units, lr):
	seq_layer = Input(shape=(seq_shape[0], seq_shape[1],))

	stage_1 = se_residual_block(seq_layer, hidden_units)
	stage_2 = se_residual_block(stage_1, hidden_units)
	stage_3 = se_residual_block(stage_2, hidden_units)

	h = GlobalAveragePooling1D()(stage_3)
	y = Dense(1, activation='sigmoid')(h)

	adam = Adam(lr=lr, clipnorm=1)
	model = Model(seq_layer, y)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
	return model

# arguments
if len(sys.argv) != 9:
	print('Usg: python {} train_x train_y valid_x valid_y test_x test_y threshold save_dir'.format(sys.argv[0]))
	exit()

data = []
for argv in sys.argv[1:-2]:
	data.append(np.load(argv))

train_x, train_y, valid_x, valid_y, test_x, test_y = data

threshold = float(sys.argv[-2])

save_dir = sys.argv[8]
if save_dir[-1] == '/':
	save_dir = save_dir[:-1]

# label y
train_y = (train_y > threshold).astype('int')
valid_y = (valid_y > threshold).astype('int')
test_y = (test_y > threshold).astype('int')

# build model and train
seq_shape = train_x.shape[1:]
model = KerasClassifier(build_fn=build_se_resnet, seq_shape=seq_shape)

param_grid = dict(batch_size=batch_size, lr=lr, hidden_units=hidden_units, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(train_x, train_y)
joblib.dump(grid.best_estimator_, '{}/grid.pkl'.format(save_dir))

print()
print("Best parameters set found:")
print()
print(grid.best_params_)
print()

# predict
print("testing model ...")
loaded_model = joblib.load('{}/grid.pkl'.format(save_dir))
predictions = loaded_model.predict(test_x, batch_size=128, verbose=1)
predictions = (predictions >= 0.5).astype('int')

test_y = test_y.reshape((-1,1))

print("Classfication report:")
print(classification_report(test_y, predictions))


