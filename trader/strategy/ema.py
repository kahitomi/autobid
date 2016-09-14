# -*- coding: utf-8 -*-
# 基于指数移动平均线的策略

import numpy as np
import talib
import matplotlib.pyplot as plt


from common import log

log = log.logger("strategy_ema")


# 长期平均和短期平均
def long_short(data_close, long_range=30, short_range=15):
	output = "calm"

	data_size = len(data_close)
	if data_size < long_range+short_range:
		log.error("The length of close data is short "+str(data_size)+"/"+str(long_range+short_range))

	data_close = np.array(data_close)

	data_EMA = talib.EMA(data_close, timeperiod=30)
	data_EMAs = talib.EMA(data_close, timeperiod=15)

	# data_differ_move = talib.EMA(data_EMAs-data_EMA, timeperiod=5)
	data_differ = data_EMAs-data_EMA


	# 计算移动
	def cal_move(_data, timeperiod=10):
		_data_move = [ _data[i]-_data[i-1] for i in range(1, len(_data)) ]
		_data_move = talib.EMA(np.array(_data_move), timeperiod=timeperiod)
		return _data_move

	data_EMA_move = cal_move(data_EMA)
	data_EMAs_move = cal_move(data_EMAs, timeperiod=20)
	data_differ_move = cal_move(data_differ)


	data_EMA = data_EMA[-15:]
	data_EMAs = data_EMAs[-15:]

	data_EMA_move = data_differ_move[-15:]
	data_EMAs_move = data_differ_move[-15:]
	data_differ_move = data_differ_move[-15:]



	current_ema = data_EMA[-1]
	current_emas = data_EMAs[-1]
	current_close = data_close[-1]
	current_differ = data_differ[-1]
	current_departure = current_close-current_emas

	current_ema_move = data_EMA_move[-1]
	current_emas_move = data_EMAs_move[-1]
	current_differ_move = data_differ_move[-1]

	long_ema_move = np.average(data_EMA_move[-3:-1])
	long_emas_move = np.average(data_EMAs_move[-3:-1])
	long_differ_move = np.average(data_differ_move[-3:-1])

	# 初始化判断参数
	move_strong_limit = 1.1

	ema_is_up = False
	ema_is_slight_up = False
	ema_is_down = False
	ema_is_slight_down = False
	ema_is_calm = False
	if long_ema_move > 0 and current_ema_move > 0:
		if current_ema_move > long_ema_move*move_strong_limit:
			ema_is_up = True
		else:
			ema_is_slight_up = True
	elif long_ema_move < 0 and current_ema_move < 0:
		if current_ema_move < long_ema_move*move_strong_limit:
			ema_is_down = True
		else:
			ema_is_slight_down = True
	else:
		ema_is_calm = True


	emas_is_up = False
	emas_is_slight_up = False
	emas_is_down = False
	emas_is_slight_down = False
	emas_is_calm = False
	if long_emas_move > 0 and current_emas_move > 0:
		if current_emas_move > long_emas_move*move_strong_limit:
			emas_is_up = True
		else:
			emas_is_slight_up = True
	elif long_emas_move < 0 and current_emas_move < 0:
		if current_emas_move < long_emas_move*move_strong_limit:
			emas_is_down = True
		else:
			emas_is_slight_down = True
	else:
		emas_is_calm = True

	differ_is_up = False
	differ_is_slight_up = False
	differ_is_down = False
	differ_is_slight_down = False
	differ_is_calm = False
	if long_differ_move > 0 and current_differ_move > 0:
		if current_differ_move > long_differ_move*move_strong_limit:
			differ_is_up = True
		else:
			differ_is_slight_up = True
	elif long_differ_move < 0 and current_differ_move < 0:
		if current_differ_move < long_differ_move*move_strong_limit:
			differ_is_down = True
		else:
			differ_is_slight_down = True
	else:
		differ_is_calm = True


	close_is_upper_ema = False
	close_is_downer_ema = False
	if current_departure > 0:
		close_is_upper_ema = True
	else:
		close_is_downer_ema = True

	close_is_far_from_ema = False
	close_is_close_from_ema = False
	if np.absolute(current_departure) > np.absolute(current_differ)*2.0:
		close_is_far_from_ema = True
	elif np.absolute(current_departure) < np.absolute(current_differ)*0.8:
		close_is_close_from_ema = True

	is_crossing = False
	next_differ = current_differ + current_differ_move*0.8
	if (current_differ > 0 and next_differ < 0) or (current_differ < 0 and next_differ > 0):
		is_crossing = True



	# 判断大趋势
	print  "ema_is_up               ", ema_is_up
	print  "ema_is_slight_up        ", ema_is_slight_up
	print  "ema_is_down             ", ema_is_down
	print  "ema_is_slight_down      ", ema_is_slight_down
	print  "ema_is_calm             ", ema_is_calm
	print  "emas_is_up              ", emas_is_up
	print  "emas_is_slight_up       ", emas_is_slight_up
	print  "emas_is_down            ", emas_is_down
	print  "emas_is_slight_down     ", emas_is_slight_down
	print  "emas_is_calm            ", emas_is_calm
	print  "differ_is_up            ", differ_is_up
	print  "differ_is_slight_up     ", differ_is_slight_up
	print  "differ_is_down          ", differ_is_down
	print  "differ_is_slight_down   ", differ_is_slight_down
	print  "differ_is_calm          ", differ_is_calm
	print  "close_is_upper_ema      ", close_is_upper_ema
	print  "close_is_downer_ema     ", close_is_downer_ema
	print  "close_is_far_from_ema   ", close_is_far_from_ema
	print  "close_is_close_from_ema ", close_is_close_from_ema
	print  "is_crossing             ", is_crossing



	if (ema_is_up or ema_is_slight_up) and (emas_is_up or emas_is_slight_up):
		# print("UP")
		# 当前close在短线之上还是之下
		if differ_is_down or differ_is_slight_down:
			if close_is_upper_ema:
				if close_is_far_from_ema:
					pass
				if close_is_close_from_ema:
					if ema_is_up and emas_is_up:
						output = "up"
			if close_is_downer_ema:
				if close_is_far_from_ema:
					if ema_is_up and emas_is_up:
						output = "up"
				if close_is_close_from_ema:
					if ema_is_up and emas_is_up:
						output = "up"
	elif (ema_is_down or ema_is_slight_down) and (emas_is_down or emas_is_slight_down):
		# print("DOWN")
		# 当前close在短线之上还是之下
		if differ_is_down or differ_is_slight_down:
			if close_is_upper_ema:
				if close_is_far_from_ema:
					if ema_is_down and emas_is_down:
						output = "down"
				if close_is_close_from_ema:
					if ema_is_down and emas_is_down:
						output = "down"
			if close_is_downer_ema:
				if close_is_far_from_ema:
					pass
				if close_is_close_from_ema:
					if ema_is_down and emas_is_down:
						output = "down"
	else:
		# print("CALM")
		pass




	print "current_ema       ", current_ema
	print "current_emas      ", current_emas
	print "current_close     ", current_close

	print "current_differ    ", current_differ
	print "current_departure ", current_departure

	print "long_ema_move and current_ema_move",long_ema_move,current_ema_move


	# plt.plot(data_EMA_move)
	# plt.plot(data_EMAs_move)
	# plt.plot(data_differ_move)
	# plt.plot(data_EMA)
	# plt.plot(data_EMAs)
	# plt.plot(data_differ)
	# plt.show()


	# log.debug(data_EMA)
	# log.debug(data_EMAs)

	return output