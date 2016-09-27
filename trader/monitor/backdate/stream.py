# -*- coding: utf-8 -*-
# 回溯策略
import csv, time
from common import config

from trader.strategy import get_strategy
from trader.make_price import price_change



# 读取流数据
def load(data_name):
	PATH_to_data = "src/data/forex/"+"BACKDATE_"+data_name+".csv"
	file_obj = open(PATH_to_data)
	reader = csv.reader(file_obj)
	data_stream = [ x for x in reader]

	return data_stream


# 用流数据回测
def simulate(data_stream, strategy = config.strategy):
	strategy = get_strategy.name(strategy)

	price_maker = price_change.maker(strategy)

	for item in data_stream:
		current_time = time.strptime(item[0], "%Y-%m-%dT%H:%M:%S.000000Z")
		ask_price = float(item[2])
		bid_price = float(item[2])

		price_maker.set_price(ask_price, bid_price)
		orders = price_maker.create_orders()

		print "============="
		print orders
		print "============="

		break