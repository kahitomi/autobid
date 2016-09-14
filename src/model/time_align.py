# -*- coding: utf-8 -*-
# 数据时间对其

import csv, datetime



# 主函数
def load(path_to_csv, period = 60):

	file_obj = open(path_to_csv)
	reader = csv.reader(file_obj)

	outputs = []

	counter = 0
	time_pointer = False
	current_time = False
	pre_time = False
	pre_item = False
	for item in reader:
		current_time = datetime.datetime.strptime(item[0], "%Y-%m-%dT%H:%M:%S.000000Z")
		# if time_pointer:
		# 	print (current_time-pre_time).seconds
		if not time_pointer or (current_time-pre_time).days > 1:
			# print "隔天"
			time_pointer = current_time


		while True:
			if time_pointer <= current_time and (time_pointer - current_time).seconds < period:
				outputs.append(item)
				time_pointer += datetime.timedelta(seconds=period)
				counter += 1
			elif time_pointer < current_time:
				outputs.append(pre_item)
				time_pointer += datetime.timedelta(seconds=period)
				counter += 1
			else:
				break
		pre_item = item
		pre_time = current_time

		if counter % 10000 == 0:
			print "TIME ALIGN COUNTER",counter

	print "TIME ALIGN COUNTER",counter
	print "Complete TIME ALIGN"

	# for i in range(1000):
	# 	print outputs[i][0]

	return outputs



if __name__ == "__main__":
	load("src/data/forex/NZD_USD_2016-07-01_2016-08-01.csv", 300)