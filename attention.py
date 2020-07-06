# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import numpy as np
from keras.layers import Input, LSTM, Dense, Permute, multiply, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# parameters
lr = 0.0001
hidden_units = 128
batch_size = 128
epochs = 100

# functions
def attention_block(inputs, time_steps):
	# input shape = (batch size, time step, input dimension)
	input_dim = int(inputs.shape[2])

	h = Permute((2, 1))(inputs)  # transpose to (input_dim, time_steps)
	h = Dense(time_steps, activation='softmax')(h)

	h_probs = Permute((2, 1), name='attention_vec')(h)  # attention vector

	output = multiply([inputs, h_probs])
	return output

def build_attention(seq_shape, hidden_units, lr):
	time_steps = seq_shape[0]

	seq_layer = Input(shape=(seq_shape[0], seq_shape[1],))

	h = LSTM(hidden_units, return_sequences=True)(seq_layer)

	# apply attention after LSTM
	attention = attention_block(h, time_steps)
	attention = Flatten()(attention)

	output = Dense(1, activation='sigmoid')(attention)

	adam = Adam(lr=lr)
	model = Model(seq_layer, output)
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
#model = build_lstm(seq_shape, hidden_units, lr)
model = KerasClassifier(build_fn=build_attention, seq_shape=seq_shape)
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


