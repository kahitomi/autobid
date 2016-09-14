# -*- coding: utf-8 -*-
# 根据数据生成买卖点

import csv
import numpy as np



# create trader points
def create(data_set):
	trade_range = 10

	slippage = 0.000010
	spread = 0.00020
	price_position = 2
	data_set_size = len(data_set)

	# 忽略多长时间内的交易
	remove_range = 1


	outputs = []
	can_profit_thread = spread + 2.0 * slippage


	for candle_point in range(data_set_size):
		trade_signal = [0.0, 0.0, 1.0] # 0: buy, 1: sell, 2: nothing
		if candle_point < (data_set_size-trade_range):
			# 遍历之后的close点
			current_price = float(data_set[candle_point][price_position])
			for i in range(remove_range,trade_range):
				this_price = float(data_set[candle_point+i][price_position])
				differ = this_price - current_price
				if differ > can_profit_thread:
					trade_signal[2] = 0.0
					trade_signal[1] = 0.0
					trade_signal[0] = 1.0
					# break
				elif differ < -can_profit_thread:
					trade_signal[2] = 0.0
					trade_signal[1] = 1.0
					trade_signal[0] = 0.0
					# break
		outputs.append(data_set[candle_point] + [trade_signal])



	# # 去掉单独的signal
	# for candle_point in range(remove_range,data_set_size-remove_range):
	# 	counter = 0
	# 	pre_sig = 2
	# 	is_remove = True
	# 	for i in range(remove_range*2):
	# 		sig = np.argmax(np.array(data_set[candle_point-remove_range+i][-1]))
	# 		is_max = False
	# 		if sig == 2:
	# 			is_max = True
	# 		elif pre_sig == sig:
	# 			is_max = True
	# 		if is_max or i == (remove_range*2-1):
	# 			if counter > remove_range:
	# 				is_remove = False
	# 			counter = 0
	# 		else:
	# 			counter += 1
	# 		pre_sig = sig
	# 	if is_remove:
	# 		data_set[candle_point][-1] = [0.0, 0.0, 1.0]


	return outputs







# from csv
def load(csv_path):
	data_set = []
	file_obj = open(csv_path)
	reader = csv.reader(file_obj)
	data_set = [ x for x in  reader]

	return create(data_set)