# -*- coding: utf-8 -*-
# 测试策略


import sys, csv, datetime, talib, time

import matplotlib.pyplot as plt
import matplotlib.finance as finance
import numpy as np


from src.model import norm





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
data_EMAs = talib.EMA(np.array(data_close), timeperiod=15)
data_EMA = talib.EMA(np.array(data_close), timeperiod=30)

data_candles = [
	[
		datetime.datetime.strptime(x[0], "%Y-%m-%dT%H:%M:%S.000000Z").minute,
		float(x[1]),
		float(x[2]),
		float(x[3]),
		float(x[4])]
	for x in data_set]

data_WILLR = talib.WILLR(np.array(data_high), np.array(data_low), np.array(data_close), timeperiod=14)
data_WILLR = -data_WILLR/100.0 # normalize
data_RSI = talib.RSI(np.array(data_close), timeperiod=14)
data_RSI = data_RSI/100.0 # normalize
data_slowk, data_slowd = talib.STOCH(np.array(data_high), np.array(data_low), np.array(data_close), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
data_slowk = data_slowk/100.0 # normalize
data_slowd = data_slowd/100.0 # normalize
data_macd, data_macdsignal, data_macdhist = talib.MACD(np.array(data_close), fastperiod=12, slowperiod=26, signalperiod=9)
data_macd = norm.tanh(data_macd, 0.0008)
data_macdsignal = norm.tanh(data_macdsignal, 0.0008)
data_macdhist = norm.tanh(data_macdhist, 0.0003)



data_close_move = [ data_close[x]-data_close[x-1] for x in range(1,len(data_close)) ]
plt.boxplot(np.absolute(np.array(data_close_move)))
ax=plt.gca()
ax.set_yticks(np.linspace(0.001,0,21)) 
plt.grid(True)
plt.show()



# 生成预测数据
input_range = 100
PATH_data_predict_emas = "src/model/forex/data_predict_emas.csv"
PATH_data_predict_ema = "src/model/forex/data_predict_ema.csv"


# # 存储
# from src.model import forex_train
# predictor = forex_train.decoder("NZD_USD_2016-05-01_2016-08-01")
# data_predict_emas = []
# data_predict_ema = []
# for i in range(input_range, len(data_close)):
# 	predict_results = predictor.decode(data_EMAs[ (i-input_range):i ], data_EMA[ (i-input_range):i ], data_WILLR[ (i-input_range):i ], data_RSI[ (i-input_range):i ], data_slowk[ (i-input_range):i ], data_slowd[ (i-input_range):i ], data_macd[ (i-input_range):i ], data_macdsignal[ (i-input_range):i ], data_macdhist[ (i-input_range):i ])
# 	# predict_results = predictor.decode(data_EMAs[ (i-input_range):i ], data_EMA[ (i-input_range):i ], data_WILLR[ (i-input_range):i ], data_RSI[ (i-input_range):i ], data_slowk[ (i-input_range):i ], data_slowd[ (i-input_range):i ], data_macd[ (i-input_range):i ], data_macdsignal[ (i-input_range):i ], data_macdhist[ (i-input_range):i ])
# 	predict_emas = np.array(predict_results)[:,:,0].reshape(-1)
# 	predict_emas = norm.tanh_decode(predict_emas, 0.001)
# 	predict_emas = predict_emas + data_EMAs[i-1]
# 	# predict_emas = predict_emas + data_close[i-1]

# 	predict_ema = np.array(predict_results)[:,:,1].reshape(-1)
# 	predict_ema = norm.tanh_decode(predict_ema, 0.001)
# 	predict_ema = predict_ema + data_EMA[i-1]

# 	data_predict_emas.append(predict_emas)
# 	data_predict_ema.append(predict_ema)
# 	# break

# file_obj = open(PATH_data_predict_emas,"wb")
# writer = csv.writer(file_obj)
# for x in data_predict_emas:
# 	writer.writerow(x)
# file_obj.close()
# file_obj = open(PATH_data_predict_ema,"wb")
# writer = csv.writer(file_obj)
# for x in data_predict_ema:
# 	writer.writerow(x)
# file_obj.close()

# 读取
file_obj = open(PATH_data_predict_emas)
reader = csv.reader(file_obj)
data_predict_emas = [ x for x in reader]
file_obj.close()
file_obj = open(PATH_data_predict_ema)
reader = csv.reader(file_obj)
data_predict_ema = [ x for x in reader]
file_obj.close()





# 生成重叠信息
data_predict_emas_overlay = [[] for _ in range(len(data_predict_emas)+5-1)]
for i in range(len(data_predict_emas)):
	p = data_predict_emas[i]
	for j in range(5):
		if len(data_predict_emas_overlay[i+j]) < 3:
			data_predict_emas_overlay[i+j].append( float(p[j]) )
data_predict_emas_avr = []
for x in data_predict_emas_overlay:
	_avr = np.average( np.array( x ) )
	data_predict_emas_avr.append(_avr)

error_list = []
trend_list = []
for i in range(input_range, len(data_close)):
	emas = float(data_EMAs[i])
	emas_pre_avr = data_predict_emas_avr[i-input_range]
	error_list.append(np.absolute(np.array(emas-emas_pre_avr)))

	prev_emas = float(data_EMAs[i-1])
	p = emas_pre_avr-prev_emas
	t = emas-prev_emas
	if (p > 0 and t > 0) or (p < 0 and t < 0):
		trend_list.append(1)
	else:
		trend_list.append(0)


print np.average(np.array(error_list))
trend_list = np.array(trend_list)
print np.sum(trend_list)/float(trend_list.size)





# from src.model import forex_trader_train
# trader = forex_trader_train.decoder("NZD_USD_2016-07-01_2016-08-01")
# data_predict_trade_point = []
# for i in range(input_range, len(data_close)):
# 	predict_results = trader.decode(data_close[ (i-input_range):i ], data_EMAs[ (i-input_range):i ], data_EMA[ (i-input_range):i ], data_WILLR[ (i-input_range):i ], data_RSI[ (i-input_range):i ], data_slowk[ (i-input_range):i ], data_slowd[ (i-input_range):i ], data_macd[ (i-input_range):i ], data_macdsignal[ (i-input_range):i ], data_macdhist[ (i-input_range):i ])
	
# 	_can_trade = np.argmax(np.array(predict_results)[0][0])
# 	if _can_trade > 0:
# 		print (np.array(predict_results)[0][0])
# 		print (np.argmax(np.array(predict_results)[0][0]))
# 		break

# 	# break






if S_NAME[0] == "ema":
	from trader.strategy import ema

	if S_NAME[1] == "long_short":
		# input_range = 70

		counter = 0
		calm_counter = 0
		correct_counter = 0
		correct_counter_s = 0
		# correct_counter = [0,0,0,0,0]
		# correct_counter_s = [0,0,0,0,0]
		for i in range(input_range, len(data_close)):
			# sig = ema.long_short(data_close[ (i-input_range):i ], data_predict_emas[i-input_range], data_predict_ema[i-input_range])











			# # try:
			# sig = [data_predict_emas[i-input_range], data_predict_ema[i-input_range]]
			# # except:
			# # 	print(len(data_predict_emas),"/",i-input_range)
			# # 	break

			# # 趋势预测测试
			# # if np.absolute(np.array(float(sig[0][0])-data_close[i])) > 0.00000:

			# if i < len(data_close)-5:
			# 	for j in range(5):
			# 		if (float(sig[0][j]) > data_EMAs[i-1+j] and data_EMAs[i+j] > data_EMAs[i-1+j]) or (float(sig[0][j]) < data_EMAs[i-1+j] and data_EMAs[i+j] < data_EMAs[i-1+j]):
			# 		# if (float(sig[0][j]) > data_close[i-1+j] and data_close[i+j] > data_close[i-1+j]) or (float(sig[0][j]) < data_close[i-1+j] and data_close[i+j] < data_close[i-1+j]):
			# 			correct_counter_s[j] += 1

			# 		# if (float(sig[1][j]) > data_EMA[i-1+j] and data_EMA[i+j] > data_EMA[i-1+j]) or (float(sig[1][j]) < data_EMA[i-1+j] and data_EMA[i+j] < data_EMA[i-1+j]):
			# 		# 	correct_counter[j] += 1

		
			# counter += 1


			# if counter % 1000 == 0:
			# 	print "#",counter
			# # break
			# continue








			# if have profit
			profit_limit = 0.00040

			show_figure = True


			sig = "anything"

			if sig == "calm":
				calm_counter += 1
				show_figure = False
			else:
				start_close = data_close[i-1]
				for _close in data_close[i:(i+5)]:
					_differ = _close-start_close
					if _differ <= -profit_limit or _differ >= profit_limit:
						correct_counter += 1
						show_figure = False
						break
					# if sig == "up" and _differ >= profit_limit:
					# 	correct_counter += 1
					# 	show_figure = False
					# 	break
					# if sig == "down" and _differ <= -profit_limit:
					# 	correct_counter += 1
					# 	show_figure = False
					# 	break


			counter += 1

			#########
			# test
			#########

			# local_false = True
			# false_counter = 0
				
			# limit = 59
			# if (counter>=limit and not local_false) or (show_figure and local_false):
			# # if show_figure:

			# 	print(sig,"==========signal")
			# 	print "*", counter

			# 	ax=plt.gca()

			# 	data_EMA = talib.EMA(np.array(data_close[(i-input_range+9):(i+20)]), timeperiod=30)
			# 	data_EMAs = talib.EMA(np.array(data_close[(i-input_range+9):(i+20)]), timeperiod=15)


			# 	plt.plot(data_EMA)
			# 	plt.plot(data_EMAs)
			# 	plt.plot(data_close[(i-input_range+9):(i+20)])

			# 	# finance.candlestick(ax, data_candles[(i-input_range):(i+20)] )

			# 	plt.grid(True)
			# 	plt.show()

			# 	if local_false:
			# 		false_counter += 1
			# 		if false_counter > 5:
			# 			break

			# if counter >= limit+5 and not local_false:
			# 	break

			# break



		print "Final chance", correct_counter, "/", counter-calm_counter
		# print "Final accurecy s", np.array(correct_counter_s)/float(counter-calm_counter)
		print "Final accurecy  ", np.array(correct_counter)/float(counter-calm_counter)





