# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd
import datetime as dt

warnings.filterwarnings('ignore')

# functions
def get_logdate(x):
	x = str(x)
	if '-' in x:
		x = dt.datetime.strptime(x, '%Y-%m-%d').strftime('%Y%m%d')
	return x

# assertions
if len(sys.argv) != 6:
	print(
		'Usg: python {} playdata.csv ad_rev.csv iap.csv total_rev.csv cpi.csv'.format(sys.argv[0]))
	exit()

# read data
playdata = pd.read_csv(sys.argv[1])
ad_rev = pd.read_csv(sys.argv[2])
iap = pd.read_csv(sys.argv[3])
total_rev = pd.read_csv(sys.argv[4])
cpi = pd.read_csv(sys.argv[5])

# convert logdate
playdata['logdate'] = playdata['logdate'].apply(get_logdate)

# join ad_clicks
ad_rev = ad_rev[['adid', 'campaign', 'age', 'clicks']]
result = pd.merge(playdata, ad_rev, on=['adid', 'age'], how='left')

# join iap
iap = iap[~iap['adid'].isnull()]
if len(iap):
	result = pd.merge(result, iap, on=['adid', 'age'], how='left')

# join total_rev
result = pd.merge(result, total_rev, on='adid', how='left')

# join cpi
result = pd.merge(result, cpi, on='campaign', how='left')

# filter data
result = result.fillna(0.0)
result['cpi'] = result['cpi'].apply(lambda x: float(x))
result = result[result['cpi'] > 0]

# calc roi
result['total_rev'] = result['total_rev'].apply(lambda x: float(x))
result['roi'] = result['total_rev'] / result['cpi']

# write
result.to_csv(sys.stdout, index=False)

