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
if len(sys.argv) != 7:
	print(
		'Usg: python {} wordcookies ios us coin 2 2019-04-01'.format(sys.argv[0]))
	exit()

if sys.argv[2] not in ['ANDROID', 'IOS', 'android', 'ios']:
	print('ERROR: invalid platfrom argument')
	exit()

# params
repoid, platform, country, main_item, input_window, installdate = sys.argv[1:]
platform = platform.upper()
country = country.upper()
input_window = int(input_window)
sdate = installdate.replace('-', '')
edate = dt.datetime.strftime(dt.datetime.strptime(installdate, '%Y-%m-%d') + dt.timedelta(days=input_window+1), '%Y%m%d')
conn = connect(host='datanode2.datawave.co.kr', port=21050)
age_limit = (input_window + 1) * 24

# find appsflyer appid
if platform == 'ANDROID':
	appid = 'com.bitmango.go.{}'.format(repoid)
else:
	url = 'http://roimon.datawave.co.kr/api/v3/apps?islive=1&fields=appid,storekey'
	appinfo = pd.read_json(url, orient='records')
	appid = appinfo[appinfo['appid']=='com.bitmango.ap.{}'.format(repoid)]['storekey'].values[0]
	appid = 'id{}'.format(appid)

# query
query = """
SELECT
	user.adid,
	af.af_id,
	af.appid,
	user.platform,
	user.country,
	user.install_date,
	user.logdate,
	user.age,
	user.sessions,
	user.session_time,
	item.item_used,
	item.item_bought,
	level.cleared_levels
FROM

(
SELECT
	UPPER(adid) as adid,
	platform,
	country,
	TO_DATE(installdatetime) AS install_date,
	logdate,
	CAST(ABS((UNIX_TIMESTAMP(date_time) - UNIX_TIMESTAMP(installdatetime)) / 3600) AS int) AS age,
	COUNT(*) AS sessions,
	SUM(cast(v1 AS float)) AS session_time
FROM
	dart3.{repoid}
WHERE
	eg = 'session' AND
	event = 'session' AND
	upper(platform) = '{platform}' AND
	upper(country) = '{country}' AND
	logdate >= '{sdate}' AND
	logdate <= '{edate}' AND
	TO_DATE(installdatetime) = '{installdate}' AND
        CAST(ABS((UNIX_TIMESTAMP(date_time) - UNIX_TIMESTAMP(installdatetime)) / 3600) AS int) < {age_limit}
GROUP BY
	adid,
	platform,
	country,
	TO_DATE(installdatetime),
	age,
	logdate,
	DATEDIFF(date_time, installdatetime)
) user

LEFT JOIN

(
SELECT
	UPPER(adid) as adid,
	CAST(ABS((UNIX_TIMESTAMP(date_time) - UNIX_TIMESTAMP(installdatetime)) / 3600) AS int) AS age,
	SUM(CASE WHEN v1 = '{main_item}' AND event = 'UncleBill_Use' THEN CAST(v2 AS INT) END) AS item_used,
	SUM(CASE WHEN v1 = '{main_item}' AND event = 'UncleBill_Buy' THEN CAST(v2 AS INT) END) AS item_bought
FROM dart3.{repoid}
WHERE
	eg = 'unclebill' AND
	event IN ('UncleBill_Use', 'UncleBill_Buy') AND
	upper(platform) = '{platform}' AND
	upper(country) = '{country}' AND
	logdate >= '{sdate}' AND
	logdate <= '{edate}' AND
	TO_DATE(installdatetime) = '{installdate}' AND
        CAST(ABS((UNIX_TIMESTAMP(date_time) - UNIX_TIMESTAMP(installdatetime)) / 3600) AS int) < {age_limit}
GROUP BY
	adid,
	age
) item

ON (user.adid = item.adid AND user.age = item.age)

LEFT JOIN

(
SELECT
	UPPER(adid) as adid,
	CAST(ABS((UNIX_TIMESTAMP(date_time) - UNIX_TIMESTAMP(installdatetime)) / 3600) AS int) AS age,
	COUNT(DISTINCT v3) as cleared_levels
FROM dart3.{repoid}
WHERE
	eg = 'clearinfo' AND
	event = 'clearinfo' AND
	v1 = '1' AND
	upper(platform) = '{platform}' AND
	upper(country) = '{country}' AND
	logdate >= '{sdate}' AND
	logdate <= '{edate}' AND
	TO_DATE(installdatetime) = '{installdate}' AND
        CAST(ABS((UNIX_TIMESTAMP(date_time) - UNIX_TIMESTAMP(installdatetime)) / 3600) AS int) < {age_limit}
GROUP BY
	adid,
	age
) level

ON (user.adid = level.adid AND user.age = level.age)

LEFT JOIN

(
SELECT
	DISTINCT
	UPPER(adid) as adid,
	af_id,
	appid
FROM appsflyer.logdata
WHERE
	event = 'install' AND
	appid = '{appid}' AND
	logdate >= '{sdate}' AND
	logdate <= '{edate}' AND
	TO_DATE(install_time) = '{installdate}'
) af

ON (user.adid = af.adid)
""".format(appid=appid, repoid=repoid, platform=platform, country=country,
	sdate=sdate, edate=edate, installdate=installdate, main_item=main_item, age_limit=age_limit)

print(query, file=sys.stderr)
df = pd.read_sql(query, conn)

# convert logdate
df['logdate'] = df['logdate'].apply(lambda x: '{}-{}-{}'.format(x[:4], x[4:6], x[6:]))

# drop null adids
df = df[~df['adid'].isnull()]

# write csv
df.to_csv(sys.stdout, index=False)








