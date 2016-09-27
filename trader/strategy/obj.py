# -*- coding: utf-8 -*-
# 策略结构


class strategy():

	def __init__(self):
		pass

	def set_price(ask, bid):
		self.ask_price = ask
		self.bid_price = bid
		self.mid_price = (ask-bid)/2.0