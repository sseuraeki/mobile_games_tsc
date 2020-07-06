# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 3:
	print('Usg: python {} keys predictions'.format(sys.argv[0]))
	exit()

df = pd.read_csv(sys.argv[1])
predictions = np.load(sys.argv[2])

df['predictions'] = predictions
df = df[['adid', 'af_id', 'appid', 'predictions']]

df.to_csv(sys.stdout, index=False)
