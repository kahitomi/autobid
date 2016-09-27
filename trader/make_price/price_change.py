# -*- coding: utf-8 -*-
# 价格变动引发的，定价
import datetime


# 定价模块
class maker():

	def __init__(self, strategy):
		self.strategy = strategy


	# 新的价格，触发功能
	def price(ask_price, bid_price, current_time = datetime.datetime.today()):
		pass