# -*- coding: utf-8 -*-
# 回溯策略


import sys, csv, datetime, talib, time

import matplotlib.pyplot as plt
import matplotlib.finance as finance
import numpy as np


from src.model import norm

from trader.monitor.backdate import stream





if len(sys.argv) < 3:
	raise ValueError("Please enter the strategy name(ema/regression) and test csv file name(NZD_USD_2016-08-01_2016-08-08)")



data_test = stream.load(sys.argv[2])
stream.simulate(data_test, sys.argv[1])