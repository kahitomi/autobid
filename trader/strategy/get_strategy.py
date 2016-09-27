# -*- coding: utf-8 -*-
# 返回策略本体

from trader.strategy import ema




def name(strategy_string):
	strategy_name_list = strategy_string.split("/")

	if strategy_name_list[0] == "ema":
		if strategy_name_list[1] == "regression":
			return ema.regression()