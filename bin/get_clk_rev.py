# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import requests
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# assertions
if len(sys.argv) != 2:
	print(
		'Usg: python {} clicks_data.csv'.format(sys.argv[0]))
	exit()

# read data
clicks = pd.read_csv(sys.argv[1])

# get parameters
dates = clicks['logdate'].unique().tolist()
dates.sort()
dates = [str(date) for date in dates]
dates = ['{}-{}-{}'.format(date[:4], date[4:6], date[6:]) for date in dates]
appid = clicks['appid'].values[0]
country = clicks['country'].values[0].lower()
roimon_token = os.environ['roimon_access_token']
if not roimon_token:
	roimon_token = os.environ['ROIMON_ACCESS_TOKEN']

# get cpc
dataset = []
df = {    # initial cpc df
	'from_': dates[0],
	'to_': dates[0],
	'account': 'all',
	'appid': 'filtered',
	'pid': 'all',
	'geo': 'filtered',
	'kpi': 'adnetpub.all.cpc.value',
	'd1': '0.0'
}
df = pd.DataFrame([df])

for date in dates:
	url = 'http://roimon.datawave.co.kr/api/v3/data?from={}&to={}'.format(date, date)
	url += '&appid={}&geo={}&kpi=adnetpub.all.cpc.value&accessToken={}'.format(appid, country, roimon_token)
	url += '&grouping=date'

	print(url, file=sys.stderr)
	res = requests.get(url)

	if len(res.content):  # update cpc only when there is data
		df = pd.DataFrame(res.json())
	else:  # exception when no data
		df = df.copy()
		df['from_'] = date
		df['to_'] = date

	dataset.append(df)

dataset = pd.concat(dataset, ignore_index=True)
cpc = dataset[['from_', 'd1']]
cpc.columns = ['logdate', 'cpc']
cpc['logdate'] = cpc['logdate'].apply(lambda x: int(''.join(x.split('-'))))

# join
result = pd.merge(clicks, cpc, on='logdate', how='left')

# calc click revenue
result['cpc'] = result['cpc'].fillna(0.0)
result['ad_rev'] = result['clicks'] * result['cpc']

# write csv
result.to_csv(sys.stdout, index=False)
