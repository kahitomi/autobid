# -*- coding: utf-8 -*-
# 基于指数移动平均线的策略

import numpy as np
import talib
import matplotlib.pyplot as plt


from common import log
from src.model import norm

from trader.strategy import obj


log = log.logger("strategy_ema")







# # 长期平均和短期平均
# def long_short(data_close, predict_emas, predict_ema, long_range=30, short_range=15):
# 	output = "calm"



# 	# data_EMAs = talib.EMA(np.array(data_close), timeperiod=15)
# 	# data_EMA = talib.EMA(np.array(data_close), timeperiod=30)
# 	# data_WILLR = talib.WILLR(np.array(data_high), np.array(data_low), np.array(data_close), timeperiod=14)
# 	# data_WILLR = -data_WILLR/100.0 # normalize
# 	# data_RSI = talib.RSI(np.array(data_close), timeperiod=14)
# 	# data_RSI = data_RSI/100.0 # normalize
# 	# data_slowk, data_slowd = talib.STOCH(np.array(data_high), np.array(data_low), np.array(data_close), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
# 	# data_slowk = data_slowk/100.0 # normalize
# 	# data_slowd = data_slowd/100.0 # normalize
# 	# data_macd, data_macdsignal, data_macdhist = talib.MACD(np.array(data_close), fastperiod=12, slowperiod=26, signalperiod=9)
# 	# data_macd = norm.tanh(data_macd, 0.0008)
# 	# data_macdsignal = norm.tanh(data_macdsignal, 0.0008)
# 	# data_macdhist = norm.tanh(data_macdhist, 0.0003)





# 	# print("======================")
# 	# print(np.array(predict_ema).shape)
# 	# print(predict_ema)
# 	# print("======================")



# 	data_size = len(data_close)
# 	if data_size < long_range+short_range:
# 		log.error("The length of close data is short "+str(data_size)+"/"+str(long_range+short_range))

# 	data_close = np.array(data_close)

# 	data_EMA = talib.EMA(data_close, timeperiod=30)
# 	data_EMAs = talib.EMA(data_close, timeperiod=15)

# 	# data_differ_move = talib.EMA(data_EMAs-data_EMA, timeperiod=5)
# 	data_differ = data_EMAs-data_EMA


# 	# 计算移动
# 	def cal_move(_data, timeperiod=10):
# 		_data_move = [ _data[i]-_data[i-1] for i in range(1, len(_data)) ]
# 		# _data_move = talib.EMA(np.array(_data_move), timeperiod=timeperiod)
# 		_data_move = np.array(_data_move)
# 		return _data_move

# 	data_EMA_move = cal_move(data_EMA)
# 	data_EMAs_move = cal_move(data_EMAs, timeperiod=20)
# 	data_differ_move = cal_move(data_differ)


# 	data_EMA = data_EMA[-15:]
# 	data_EMAs = data_EMAs[-15:]

# 	data_EMA_move = data_EMA_move[-15:]
# 	data_EMAs_move = data_EMAs_move[-15:]
# 	data_differ_move = data_differ_move[-15:]


# 	# print data_EMA[-2:]
# 	# print data_EMA_move[-2:]



# 	current_ema = data_EMA[-1]
# 	current_emas = data_EMAs[-1]
# 	current_close = data_close[-1]
# 	current_differ = data_differ[-1]
# 	current_departure = current_close-current_emas

# 	pre_ema = data_EMA[-2]
# 	pre_emas = data_EMAs[-2]
# 	pre_close = data_close[-2]

# 	current_ema_move = data_EMA_move[-1]
# 	current_emas_move = data_EMAs_move[-1]
# 	current_differ_move = data_differ_move[-1]
# 	current_close_move = current_close-data_close[-2]

# 	pre_ema_move = data_EMA_move[-1]
# 	pre_emas_move = data_EMAs_move[-1]
# 	pre_differ_move = data_differ_move[-1]

# 	long_ema_move = np.average(data_EMA_move[-5:-1])
# 	long_emas_move = np.average(data_EMAs_move[-5:-1])
# 	long_differ_move = np.average(data_differ_move[-5:-1])

# 	# 初始化判断参数
# 	move_strong_limit = 1.1

# 	ema_is_up = False
# 	ema_is_slight_up = False
# 	ema_is_turn_up = False
# 	ema_is_down = False
# 	ema_is_slight_down = False
# 	ema_is_turn_down = False
# 	ema_is_calm = False
# 	if long_ema_move > 0 and current_ema_move > 0:
# 		if current_ema_move > long_ema_move*move_strong_limit:
# 			ema_is_up = True
# 		else:
# 			ema_is_slight_up = True
# 	elif long_ema_move < 0 and current_ema_move < 0:
# 		if current_ema_move < long_ema_move*move_strong_limit:
# 			ema_is_down = True
# 		else:
# 			ema_is_slight_down = True
# 	elif pre_ema_move < 0 and current_ema_move > 0:
# 		if np.absolute(long_ema_move)*move_strong_limit < np.absolute(current_ema_move):
# 			ema_is_turn_up = True
# 		else:
# 			ema_is_calm = True
# 	elif pre_ema_move > 0 and current_ema_move < 0:
# 		if np.absolute(long_ema_move)*move_strong_limit < np.absolute(current_ema_move):
# 			ema_is_turn_down = True
# 		else:
# 			ema_is_calm = True
# 	else:
# 		ema_is_calm = True


# 	emas_is_up = False
# 	emas_is_slight_up = False
# 	emas_is_turn_up = False
# 	emas_is_down = False
# 	emas_is_slight_down = False
# 	emas_is_turn_down = False
# 	emas_is_calm = False
# 	if long_emas_move > 0 and current_emas_move > 0:
# 		if current_emas_move > long_emas_move*move_strong_limit:
# 			emas_is_up = True
# 		else:
# 			emas_is_slight_up = True
# 	elif long_emas_move < 0 and current_emas_move < 0:
# 		if current_emas_move < long_emas_move*move_strong_limit:
# 			emas_is_down = True
# 		else:
# 			emas_is_slight_down = True
# 	elif pre_emas_move < 0 and current_emas_move > 0:
# 		if np.absolute(long_emas_move)*move_strong_limit < np.absolute(current_emas_move):
# 			emas_is_turn_up = True
# 		else:
# 			emas_is_calm = True
# 	elif pre_emas_move > 0 and current_emas_move < 0:
# 		if np.absolute(long_emas_move)*move_strong_limit < np.absolute(current_emas_move):
# 			emas_is_turn_down = True
# 		else:
# 			emas_is_calm = True
# 	else:
# 		emas_is_calm = True

# 	differ_is_up = False
# 	differ_is_slight_up = False
# 	differ_is_turn_up = False
# 	differ_is_down = False
# 	differ_is_slight_down = False
# 	differ_is_turn_down = False
# 	differ_is_calm = False
# 	if long_differ_move > 0 and current_differ_move > 0:
# 		if current_differ_move > long_differ_move*move_strong_limit:
# 			differ_is_up = True
# 		else:
# 			differ_is_slight_up = True
# 	elif long_differ_move < 0 and current_differ_move < 0:
# 		if current_differ_move < long_differ_move*move_strong_limit:
# 			differ_is_down = True
# 		else:
# 			differ_is_slight_down = True
# 	elif pre_differ_move < 0 and current_differ_move > 0:
# 		if np.absolute(long_differ_move)*move_strong_limit < np.absolute(current_differ_move):
# 			differ_is_turn_up = True
# 		else:
# 			differ_is_calm = True
# 	elif pre_differ_move > 0 and current_differ_move < 0:
# 		if np.absolute(long_differ_move)*move_strong_limit < np.absolute(current_differ_move):
# 			differ_is_turn_down = True
# 		else:
# 			differ_is_calm = True
# 	else:
# 		differ_is_calm = True


# 	close_is_upper_ema = False
# 	close_is_downer_ema = False
# 	close_is_in_ema = False
# 	if current_departure > 0 and current_close > current_ema:
# 		close_is_upper_ema = True
# 	elif current_departure < 0 and current_close < current_ema:
# 		close_is_downer_ema = True
# 	else:
# 		close_is_in_ema = True

# 	close_is_far_from_ema = False
# 	close_is_close_from_ema = False
# 	if np.absolute(current_departure) > np.absolute(current_differ)*2.0:
# 		close_is_far_from_ema = True
# 	elif np.absolute(current_departure) < np.absolute(current_differ)*0.8:
# 		close_is_close_from_ema = True

# 	is_crossing = False
# 	next_differ = current_differ + current_differ_move*0.8
# 	if (current_differ > 0 and next_differ < 0) or (current_differ < 0 and next_differ > 0):
# 		is_crossing = True
# 	# if next_differ < 0.00020:
# 	# 	is_crossing = True



# 	# 预测部分
# 	trend_ema_list = []
# 	trend_emas_list = []
# 	_pre_ema = current_ema
# 	for x in predict_ema:
# 		if x >= _pre_ema:
# 			trend_ema_list.append(1)
# 		else:
# 			trend_ema_list.append(-1)
# 		_pre_ema = x
# 	_pre_emas = current_emas
# 	for x in predict_emas:
# 		if x >= _pre_emas:
# 			trend_emas_list.append(1)
# 		else:
# 			trend_emas_list.append(-1)
# 		_pre_emas = x

# 	trend_ema_up = True
# 	trend_ema_down = True
# 	for x in trend_ema_list[2:]:
# 		if x > 0:
# 			trend_ema_down = False
# 		elif x < 0:
# 			trend_ema_up = False
# 	trend_emas_up = True
# 	trend_emas_down = True
# 	for x in trend_emas_list[2:]:
# 		if x > 0:
# 			trend_emas_down = False
# 		elif x < 0:
# 			trend_emas_up = False

# 	# trend_ema_all_up = False
# 	# trend_ema_all_down = False
# 	# for x in predict_ema[2:]:
# 	# 	if x > 0:
# 	# 		trend_ema_down = False
# 	# 	elif x < 0:
# 	# 		trend_ema_up = False





# 	# # 判断大趋势
# 	# if ema_is_up: print "ema_is_up"
# 	# if ema_is_slight_up: print "ema_is_slight_up"
# 	# if ema_is_turn_up: print "ema_is_turn_up"
# 	# if ema_is_down: print "ema_is_down"
# 	# if ema_is_slight_down: print "ema_is_slight_down"
# 	# if ema_is_turn_down: print "ema_is_turn_down"
# 	# if ema_is_calm: print "ema_is_calm"
# 	# if emas_is_up: print "emas_is_up"
# 	# if emas_is_slight_up: print "emas_is_slight_up"
# 	# if emas_is_turn_up: print "emas_is_turn_up"
# 	# if emas_is_down: print "emas_is_down"
# 	# if emas_is_slight_down: print "emas_is_slight_down"
# 	# if emas_is_turn_down: print "emas_is_turn_down"
# 	# if emas_is_calm: print "emas_is_calm"
# 	# if differ_is_up: print "differ_is_up"
# 	# if differ_is_slight_up: print "differ_is_slight_up"
# 	# if differ_is_turn_up: print "differ_is_turn_up"
# 	# if differ_is_down: print "differ_is_down"
# 	# if differ_is_slight_down: print "differ_is_slight_down"
# 	# if differ_is_turn_down: print "differ_is_turn_down"
# 	# if differ_is_calm: print "differ_is_calm"

# 	# if close_is_upper_ema: print "close_is_upper_ema"
# 	# if close_is_downer_ema: print "close_is_downer_ema"
# 	# if close_is_in_ema: print "close_is_in_ema"
# 	# if close_is_far_from_ema: print "close_is_far_from_ema"
# 	# if close_is_close_from_ema: print "close_is_close_from_ema"
# 	# if is_crossing: print "is_crossing"




# 	# if (ema_is_up or ema_is_slight_up) and (emas_is_up or emas_is_slight_up):
# 	# 	# print("UP")
# 	# 	# 当前close在短线之上还是之下
# 	# 	if differ_is_down or differ_is_slight_down:
# 	# 		if close_is_upper_ema:
# 	# 			# 之上
# 	# 			if close_is_far_from_ema:
# 	# 				pass
# 	# 			if close_is_close_from_ema:
# 	# 				pass
# 	# 		if close_is_downer_ema:
# 	# 			# 之下
# 	# 			if close_is_far_from_ema:
# 	# 				if ema_is_up and emas_is_up:
# 	# 					output = "up"
# 	# 			if close_is_close_from_ema:
# 	# 				if ema_is_up and emas_is_up:
# 	# 					output = "up"
# 	# elif (ema_is_down or ema_is_slight_down) and (emas_is_down or emas_is_slight_down):
# 	# 	# print("DOWN")
# 	# 	# 当前close在短线之上还是之下
# 	# 	if differ_is_down or differ_is_slight_down:
# 	# 		if close_is_upper_ema:
# 	# 			# 之上
# 	# 			if close_is_far_from_ema:
# 	# 				if ema_is_down and emas_is_down:
# 	# 					output = "down"
# 	# 			if close_is_close_from_ema:
# 	# 				if ema_is_down and emas_is_down:
# 	# 					output = "down"
# 	# 		if close_is_downer_ema:
# 	# 			# 之下
# 	# 			if close_is_far_from_ema:
# 	# 				pass
# 	# 			if close_is_close_from_ema:
# 	# 				pass


# 	if trend_emas_up:
# 		output = "up"
# 	if trend_emas_down:
# 		output = "down"
# 		# print "Down"

# 	# if close_is_in_ema:
# 	# 	# 转折判断
# 	# 	just_in = True
# 	# 	for i in range(5):
# 	# 		if (data_close[-(i+2)] < data_EMA[-(i+2)] and data_close[-(i+2)] > data_EMAs[-(i+2)]) or (data_close[-(i+2)] > data_EMA[-(i+2)] and data_close[-(i+2)] < data_EMAs[-(i+2)]):
# 	# 			just_in = False
# 	# 			break
# 	# 	if just_in:
# 	# 		# 刚刚突入
# 	# 		if pre_ema_move > 0 and pre_emas_move < 0:
# 	# 			if trend_ema_up:
# 	# 				output = "up"
# 	# 			if trend_ema_down:
# 	# 				output = "down"


# 	# 			# print current_differ_move*10.0 ,"-", current_differ
# 	# 			# if np.absolute(current_differ_move*10.0) >= np.absolute(current_differ):
# 	# 			# 	if trend_ema_down:
# 	# 			# 		output = "down"
# 	# 			# else:
# 	# 			# 	if trend_ema_up:
# 	# 			# 		output = "up"


# 		# elif (ema_is_down or ema_is_slight_down) and (emas_is_up or emas_is_slight_up):
# 		# 	output = "down"


# 	# elif ema_is_slight_up and emas_is_calm and differ_is_down and close_is_in_ema:
# 	# 	if current_close_move > 0:
# 	# 		output = "up"
# 	# 	else:
# 	# 		output = "down"


# 	# elif ema_is_turn_up and emas_is_turn_up:
# 	# 	# 转上
# 	# 	if not close_is_far_from_ema:
# 	# 		output = "up"
# 	# elif ema_is_turn_up and emas_is_turn_up:
# 	# 	# 转下
# 	# 	if not close_is_far_from_ema:
# 	# 		output = "down"
# 	# else:
# 	# 	# print("CALM")
# 	# 	pass




# 	# print "current_ema       ", current_ema
# 	# print "current_emas      ", current_emas
# 	# print "current_close     ", current_close

# 	# print "current_differ    ", current_differ
# 	# print "current_departure ", current_departure

# 	# print "long_ema_move and current_ema_move",long_ema_move,current_ema_move

# 	# print "Pre emas", predict_emas
# 	# print "Pre ema", predict_ema

# 	# print "trend_ema_list", trend_ema_list
# 	# print "trend_emas_list", trend_emas_list



# 	# plt.plot(data_EMA_move)
# 	# plt.plot(data_EMAs_move)
# 	# plt.plot(data_differ_move)
# 	# plt.plot(data_EMA)
# 	# plt.plot(data_EMAs)
# 	# plt.plot(data_differ)
# 	# plt.show()


# 	# log.debug(data_EMA)
# 	# log.debug(data_EMAs)

# 	return output





# 回归平均线策略
class regression(obj.strategy):
	
	def undefined(self):
		pass