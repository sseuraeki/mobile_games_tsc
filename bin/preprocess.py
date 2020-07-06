# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# functions
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()
	return None

def one_hot(dataset, column_name):
	uniques = dataset[column_name].unique()
	newcols = []
	for unique in uniques:
		newcol = '{}_{}'.format(column_name, str(unique))
		newcols.append(newcol)
		dataset[newcol] = dataset[column_name].apply(lambda x: 1.0 if x == unique else 0.0)
	del dataset[column_name]
	return dataset, newcols

# arguments
if len(sys.argv) != 5:
	print('Usg: python {} data.csv output_keys.csv output_series.npy output_targets.npy'.format(sys.argv[0]))
	exit()

# parameters
datafile = sys.argv[1]
output_keys = sys.argv[2]
output_series = sys.argv[3]
output_targets = sys.argv[4]

# read file
df = pd.read_csv(datafile)
df = df.fillna(0.0)
df = df[df['adid'] != '0.0']

# af_ids
appid = df['appid'].unique()[0]
af_ids = df[['adid', 'af_id']].drop_duplicates()

# remove unneeded cols
del df['af_id'], df['appid'], df['platform'], df['country'], df['install_date'], df['logdate']
del df['campaign'], df['cpi'], df['total_rev']

# get loop lists
adids = df['adid'].unique().tolist()
ages = df['age'].unique().tolist()
ages.sort()

# result keys
keys = pd.DataFrame(adids, columns=['adid'])
keys = pd.merge(keys, af_ids, on='adid', how='left')
keys['appid'] = appid

# preprocess
series = []
targets = []

for i in range(len(adids)):
	printProgress(i, len(adids))
	adid = adids[i]
	tmp = df[df['adid']==adid].copy()
	#targets.append(tmp['total_rev'].values[0])
	targets.append(tmp['roi'].values[0])
	del tmp['roi']

	if len(tmp) == len(ages):  # if all ages present
		tmp = tmp.sort_values('age')
		del tmp['adid'], tmp['age']
		series.append(tmp.values)
	else:  # if there are any missing ages
		adid_series = []
		for age in ages:
			adid_tmp = tmp[tmp['age']==age]
			del adid_tmp['adid'], adid_tmp['age']
			if len(adid_tmp):
				adid_series.append(adid_tmp.values[0])
			else:
				adid_series.append(np.zeros((len(adid_tmp.columns))))

		series.append(adid_series)

series = np.array(series)
targets = np.array(targets)

# write
keys.to_csv(output_keys, index=False)
np.save(output_series, series)
np.save(output_targets, targets)


