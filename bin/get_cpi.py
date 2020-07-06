# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import warnings
import requests
import pandas as pd

warnings.filterwarnings('ignore')

# assertions
if len(sys.argv) != 5:
	print(
		'Usg: python {} wordcookies android us 2020-02-26'.format(sys.argv[0]))
	exit()

# params
accessToken = os.environ['roimon_access_token']
if not accessToken:
	accessToken = os.environ['ROIMON_ACCESS_TOKEN']

repoid = sys.argv[1]
platform = sys.argv[2]
geo = sys.argv[3]
date = sys.argv[4]
grouping = 'date,c1'
kpi = 'campaign.all.spend.value,campaign.all.install.value'

if platform == 'android':
	appid = 'com.bitmango.go.' + repoid
elif platform == 'ios':
	appid = 'com.bitmango.ap.' + repoid
else:
	print('ERROR: Wrong platform name "{}"'.format(platform))
	exit()

# get
url = 'http://roimon.datawave.co.kr/api/v3/data?from={date}&to={date}'.format(date=date)
url += '&appid={appid}&geo={geo}&grouping={grouping}&kpi={kpi}&accessToken={accessToken}'.format(
	appid=appid, geo=geo, grouping=grouping, kpi=kpi, accessToken=accessToken)

print(url, file=sys.stderr)
res = requests.get(url)
df = pd.DataFrame(res.json())

# spread df
spend = df[df['kpi'] == 'campaign.all.spend.value'].copy()
spend['campaign'] = spend['c1']
spend['spend'] = spend['d1'].apply(lambda x: float(x))
spend = spend[['campaign', 'spend']]

install = df[df['kpi'] == 'campaign.all.install.value'].copy()
install['campaign'] = install['c1']
install['install'] = install['d1'].apply(lambda x: float(x))
install = install[['campaign', 'install']]

spread = pd.merge(spend, install, on='campaign', how='outer')
spread = spread.fillna(0.0)
spread = spread[spread['install'] > 0]
spread['cpi'] = spread['spend'] / spread['install']
spread = spread[['campaign', 'cpi']]

# write
spread.to_csv(sys.stdout, index=False)



