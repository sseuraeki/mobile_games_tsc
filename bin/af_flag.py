# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from gevent import monkey
monkey.patch_all()

import sys
import json
import requests
import pandas as pd
from datetime import datetime as dt

import gevent

# functions
def get_sample(pd_series, event_name):
	sample = {
		'appsflyer_id': pd_series['af_id'],
		'eventName': event_name,
		'af_events_api': 'true'
	}
	return sample

def post(url, headers, data):
	res = requests.post(url, headers=headers, data=data)
	gevent.sleep(0)
	if res.status_code != 200:
		print('ERROR code:', res.status_code, file=sys.stderr)
		print('ERROR msg:', res.content, file=sys.stderr)
		print('ERROR sample:', file=sys.stderr)
		print(data, file=sys.stderr)
		print('')

# params
if len(sys.argv) != 4:
	print('Usg: python {} data.csv event_name dev_key'.format(sys.argv[0]))
	exit()

df = pd.read_csv(sys.argv[1])
appid = df['appid'].unique()[0]
event_name = sys.argv[2]
dev_key = sys.argv[3]

# drop null af_ids
df = df[df['af_id']!='0.0']

# filter those need to be flagged
df = df[df['predictions']==1.0]

# make url
# https://support.appsflyer.com/hc/en-us/articles/207034486-Server-to-server-events-API#setup
url = 'https://api2.appsflyer.com/inappevent/{app_id}'.format(app_id=appid)

# header
headers = {'authentication': dev_key}

# post
batch_size = 100
print('Posting Appsflyer events ...', file=sys.stderr)

for i in range(0, len(df), batch_size):
	# get 100 samples
	batch_start = i
	batch_end = i + batch_size
	if batch_end > len(df):
		batch = df.iloc[batch_start:]
	else:
		batch = df.iloc[batch_start:batch_end]

	threads = []
	for j in range(len(batch)):
		tmp = batch.iloc[j]

		sample = get_sample(tmp, event_name)
		data = json.dumps(sample)
		threads.append(gevent.spawn(post, url, headers, data))

	gevent.joinall(threads)

