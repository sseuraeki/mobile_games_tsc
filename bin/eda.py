# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# arguments
if len(sys.argv) != 2:
	print(
		'Usg: python {} "data/wordcookies.ios.us/*/result.csv"'.format(
			sys.argv[0]))
	exit()

# load
dataset = []
for f in glob.glob(sys.argv[1]):
	print(f)
	dataset.append(pd.read_csv(f))
df = pd.concat(dataset, ignore_index=True)

# fillna
df = df.fillna(0.0)

# eda
df2 = df[['adid', 'roi']].drop_duplicates()
print(df2['roi'].describe())

print('Top 10%:', df2['roi'].quantile(0.9))


# count
#print(len(df[df['total_rev'] >= 1.10]))


# top_10_percent