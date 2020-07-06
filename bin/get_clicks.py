# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import warnings
import pandas as pd
import datetime as dt
from impala.dbapi import connect

warnings.filterwarnings('ignore')

# assertions
if len(sys.argv) != 6:
	print(
		'Usg: python {} wordcookies ios us 30 2019-04-01'.format(sys.argv[0]))
	exit()

if sys.argv[2] not in ['ANDROID', 'IOS', 'android', 'ios']:
	print('ERROR: invalid platfrom argument')
	exit()

# params
repoid, platform, country, target_window, installdate = sys.argv[1:]
platform = platform.upper()
country = country.upper()
target_window = int(target_window)
sdate = installdate.replace('-', '')
edate = dt.datetime.strftime(dt.datetime.strptime(installdate, '%Y-%m-%d') + dt.timedelta(days=target_window+1), '%Y%m%d')
age_limit = (target_window + 1) * 24

if platform == 'ANDROID':
	appid = 'com.bitmango.go.{}'.format(repoid)
else:
	url = 'http://roimon.datawave.co.kr/api/v3/apps?islive=1&fields=appid,storekey'
	appinfo = pd.read_json(url, orient='records')
	appid = appinfo[appinfo['appid']=='com.bitmango.ap.{}'.format(repoid)]['storekey'].values[0]
	appid = 'id{}'.format(appid)

conn = connect(host='datanode2.datawave.co.kr', port=21050)

# query
query = """
SELECT
	UPPER(adid) as adid,
	c as campaign,
	logdate,
	CAST(ABS((UNIX_TIMESTAMP(event_time) - UNIX_TIMESTAMP(install_time)) / 3600) AS int) AS age,
	count(*) as clicks
FROM appsflyer.logdata
WHERE
	appid = '{appid}' AND
	upper(geo) = '{country}' AND
	logdate >= '{sdate}' AND
	logdate <= '{edate}' AND
	TO_DATE(install_time) = '{installdate}' AND
	event = 'adclick' AND
        CAST(ABS((UNIX_TIMESTAMP(event_time) - UNIX_TIMESTAMP(install_time)) / 3600) AS int) < {age_limit}
GROUP BY
	adid, c, logdate, age
""".format(appid=appid, country=country, sdate=sdate, edate=edate, installdate=installdate, age_limit=age_limit)

df = pd.read_sql(query, conn)
df['adid'] = df['adid'].apply(lambda x: x.upper())
df['appid'] = 'com.bitmango.go.{}'.format(repoid) if platform == 'ANDROID' else 'com.bitmango.ap.{}'.format(repoid)
df['country'] = country

df.to_csv(sys.stdout, index=False)





