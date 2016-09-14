# -*- coding: utf-8 -*-
# 日志统计方法

import logging

# from mlogging import TimedRotatingFileHandler_MP as TimedRotatingFileHandler

root_log = logging.getLogger()
root_log.setLevel(logging.DEBUG)

# # 创建一个handler，用于写入日志文件
# fh = TimedRotatingFileHandler("common/log/das.log", when='midnight', interval=1, backupCount=0, encoding=None, delay=False, utc=False)
# fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
# root_log.addHandler(fh)
root_log.addHandler(ch)


# 创建一个log对象
def logger(name):
	# 创建一个logger
	_logger = logging.getLogger(name)
	_logger.setLevel(logging.DEBUG)

	return _logger

	# # 记录一条日志
	# logger.info('example')