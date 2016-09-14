# -*- coding: utf-8 -*-
# 测试策略


import sys, csv, datetime, talib, time

import matplotlib.pyplot as plt
import matplotlib.finance as finance
import numpy as np





if len(sys.argv) < 3:
	raise ValueError("Please enter the strategy name(ema/long_short) and test csv file name(xxxxx.csv)")

S_NAME = sys.argv[1].split("/")
SOURCE_PATH = "src/data/forex/"
TEST_DATA_PATH = SOURCE_PATH + sys.argv[2]


# 读取测试数据
data_set = []
file_obj = open(TEST_DATA_PATH)
reader = csv.reader(file_obj)
data_set = [ x for x in reader]

data_open = [ float(x[1]) for x in data_set ]
data_close = [ float(x[2]) for x in data_set ]
data_high = [ float(x[3]) for x in data_set ]
data_low = [ float(x[4]) for x in data_set ]

data_candles = [
	[
		datetime.datetime.strptime(x[0], "%Y-%m-%dT%H:%M:%S.000000Z").minute,
		float(x[1]),
		float(x[2]),
		float(x[3]),
		float(x[4])]
	for x in data_set]



if S_NAME[0] == "ema":
	from trader.strategy import ema

	if S_NAME[1] == "long_short":
		input_range = 50

		counter = 0
		for i in range(input_range, len(data_close)):
			sig = ema.long_short(data_close[ (i-input_range):i ])
			print(sig,"==========signal")

			ax=plt.gca()

			data_EMA = talib.EMA(np.array(data_close[(i-input_range+9):(i+20)]), timeperiod=30)
			data_EMAs = talib.EMA(np.array(data_close[(i-input_range+9):(i+20)]), timeperiod=15)

			limit = 530
			if counter >= limit:

				plt.plot(data_EMA)
				plt.plot(data_EMAs)
				plt.plot(data_close[(i-input_range+9):(i+20)])

				# finance.candlestick(ax, data_candles[(i-input_range):(i+20)] )

				plt.grid(True)
				plt.show()

			counter += 1
			if counter >= limit+5:
				break




