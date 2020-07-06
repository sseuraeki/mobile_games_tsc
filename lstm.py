# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# parameters for grid search
batch_size = [128, 256, 512]
lr = [1e-3, 1e-4]
hidden_units = [64, 128, 256]
epochs = [10]

# functions
def build_lstm(seq_shape, hidden_units, lr):
	seq_layer = Input(shape=(seq_shape[0], seq_shape[1],))

	h = LSTM(hidden_units, return_sequences=True)(seq_layer)
	h = LSTM(hidden_units)(h)

	y = Dense(1, activation='sigmoid')(h)

	adam = Adam(lr=lr)
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
model = KerasClassifier(build_fn=build_lstm, seq_shape=seq_shape)

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
predictions = loaded_model.predict(valid_x, batch_size=128, verbose=1)
predictions = (predictions >= 0.5).astype('int')

valid_y = valid_y.reshape((-1,1))

print("Classfication report:")
print(classification_report(valid_y, predictions))


