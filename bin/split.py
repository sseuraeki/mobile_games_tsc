# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# arguments
if len(sys.argv) != 6:
	print(
		'Usg: python {} "data/wordcookies.ios.us/*/result.csv" 0.39 train.csv valid.csv test.csv'.format(
			sys.argv[0]))
	exit()

datafiles = sys.argv[1]
threshold = float(sys.argv[2])
train_path = sys.argv[3]
valid_path = sys.argv[4]
test_path = sys.argv[5]

# set seed
np.random.seed(123)

# read data
dataset = []
for f in glob.glob(datafiles):
	dataset.append(pd.read_csv(f, dtype={'adid':str, 'af_id':str, 'appid':str}))
df = pd.concat(dataset, ignore_index=True)
df = df[~df['adid'].isnull()]
df = df.fillna(0.0)

# standardize
cols_to_stand = [
	'sessions', 'session_time', 'item_used', 'item_bought', 'cleared_levels', 'clicks',
	'iap', 'total_rev', 'cpi'
	]

#df[cols_to_stand] = StandardScaler().fit_transform(df[cols_to_stand])

# split by threshold - target1 is the smaller
#above = df[df['total_rev']>threshold]
#below = df[df['total_rev']<=threshold]
above = df[df['roi']>threshold]
below = df[df['roi']<=threshold]

if len(above) < len(below):
	target1 = above.copy()
	target0 = below.copy()
else:
	target1 = below.copy()
	target0 = above.copy()

# sample balanced trainset
target1_adids = target1['adid'].unique().tolist()
target0_adids = target0['adid'].unique().tolist()

n_t1_adids = len(target1_adids)
n_train_t1 = int(n_t1_adids * 0.8)
train_t1_adids = np.random.choice(target1_adids, n_train_t1, replace=False).tolist()
train_t0_adids = np.random.choice(target0_adids, len(train_t1_adids), replace=False).tolist()
train_adids = train_t1_adids + train_t0_adids
trainset = df[df['adid'].isin(train_adids)].sample(frac=1)  # shuffling

# sample validset
for adid in train_t1_adids:
	target1_adids.remove(adid)
for adid in train_t0_adids:
	target0_adids.remove(adid)

n_valid_t1 = int(n_t1_adids * 0.1)
valid_t1_adids = np.random.choice(target1_adids, n_valid_t1, replace=False).tolist()
valid_t0_adids = np.random.choice(target0_adids, len(valid_t1_adids), replace=False).tolist()
valid_adids = valid_t1_adids + valid_t0_adids
validset = df[df['adid'].isin(valid_adids)].sample(frac=1)

# the rest will be testset
sampled_adids = train_adids + valid_adids
testset = df[~df['adid'].isin(sampled_adids)].sample(frac=1)

# write files
trainset.to_csv(train_path, index=False)
validset.to_csv(valid_path, index=False)
testset.to_csv(test_path, index=False)









