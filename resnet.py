import sys
import pickle
import numpy as np
from keras.layers import Input, Conv1D, Dense
from keras.layers import BatchNormalization, Activation, add
from keras.layers import GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.models import Model, model_from_json
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
def build_resnet(seq_shape, hidden_units, lr):
	seq_layer = Input(shape=(seq_shape[0], seq_shape[1],))

	# block 1
	b1 = Conv1D(filters=hidden_units, kernel_size=8, padding='same')(seq_layer)
	b1 = BatchNormalization()(b1)
	b1 = Activation('relu')(b1)

	b1 = Conv1D(filters=hidden_units, kernel_size=5, padding='same')(b1)
	b1 = BatchNormalization()(b1)
	b1 = Activation('relu')(b1)

	b1 = Conv1D(filters=hidden_units, kernel_size=3, padding='same')(b1)
	b1 = BatchNormalization()(b1)

	shortcut = Conv1D(filters=hidden_units, kernel_size=1, padding='same')(seq_layer)
	shortcut = BatchNormalization()(shortcut)

	b1 = add([shortcut, b1])
	b1 = Activation('relu')(b1)

	# block 2
	b2 = Conv1D(filters=hidden_units*2, kernel_size=8, padding='same')(b1)
	b2 = BatchNormalization()(b2)
	b2 = Activation('relu')(b2)

	b2 = Conv1D(filters=hidden_units*2, kernel_size=5, padding='same')(b2)
	b2 = BatchNormalization()(b2)
	b2 = Activation('relu')(b2)

	b2 = Conv1D(filters=hidden_units*2, kernel_size=3, padding='same')(b2)
	b2 = BatchNormalization()(b2)

	shortcut = Conv1D(filters=hidden_units*2, kernel_size=1, padding='same')(b1)
	shortcut = BatchNormalization()(shortcut)

	b2 = add([shortcut, b2])
	b2 = Activation('relu')(b2)

	# block 3
	b3 = Conv1D(filters=hidden_units*2, kernel_size=8, padding='same')(b2)
	b3 = BatchNormalization()(b3)
	b3 = Activation('relu')(b3)

	b3 = Conv1D(filters=hidden_units*2, kernel_size=5, padding='same')(b3)
	b3 = BatchNormalization()(b3)
	b3 = Activation('relu')(b3)

	b3 = Conv1D(filters=hidden_units*2, kernel_size=3, padding='same')(b3)
	b3 = BatchNormalization()(b3)

	shortcut = BatchNormalization()(b2)

	b3 = add([shortcut, b3])
	b3 = Activation('relu')(b3)

	# output
	pooling = GlobalAveragePooling1D()(b3)
	y = Dense(1, activation='sigmoid')(pooling)

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
model = KerasClassifier(build_fn=build_resnet, seq_shape=seq_shape)

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




