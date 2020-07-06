# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd
import datetime as dt

warnings.filterwarnings('ignore')

# assertions
if len(sys.argv) != 3:
	print(
		'Usg: python {} ad_rev.csv iap.csv'.format(sys.argv[0]))
	exit()

# read data
ad_rev = pd.read_csv(sys.argv[1])
iap = pd.read_csv(sys.argv[2])

# no data exceptions
if len(ad_rev) == 0:
	ad_rev = pd.DataFrame([{'adid': '0',
			'age': 0,
			'clicks': 0,
			'appid': 0,
			'country': 0,
			'cpc': 0,
			'ad_rev': 0},
			{'adid': '1',
			'age': 0,
			'clicks': 0,
			'appid': 0,
			'country': 0,
			'cpc': 0,
			'ad_rev': 0}])
if len(iap) == 0:
	iap = pd.DataFrame([{'adid': '0',
			'age': 0,
			'iap': 0},
			{'adid': '1',
			'age': 0,
			'iap': 0}])

# group by sum
ad_rev = ad_rev.groupby('adid', as_index=False)['ad_rev'].sum()
iap = iap.groupby('adid', as_index=False)['iap'].sum()

# join
result = pd.merge(ad_rev, iap, on='adid', how='outer')
result = result.fillna(0.0)
result = result.groupby('adid', as_index=False)[['ad_rev', 'iap']].sum()

# drop temporary adids
result = result[(result['adid'] != '0') & (result['adid'] != '1')]

# calc total rev
result = result.fillna(0.0)
result['total_rev'] = result['ad_rev'] + result['iap']
result = result[['adid', 'total_rev']]

# write
result.to_csv(sys.stdout, index=False)


