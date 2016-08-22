# -*- coding: utf-8 -*-
# 导入下载的外汇csv文件，生成用于训练的数据文件

import sys, csv, datetime




if len(sys.argv) < 2:
	raise ValueError("Please enter the forex csv file path")

csv_path = sys.argv[1]

if len(sys.argv) < 3:
	output_file_name = csv_path.split("/")[-1]
else:
	output_file_name = sys.argv[2]

output_path = "src/data/forex/"



PRICE_SCALE = 15
SECOND_VOLUME = 2



reader = csv.reader(open(csv_path))
pointer = [0, 0, 0, 0, 0, 0]
bid_price_block = []
ask_price_block = []
m_second_block = []
second_block_data = []

min_length = 1000
sum_length = 0
length_counter = 0

row_counter = 0
sec_counter = 0

over_price_counter = 0

max_price = 0.0
min_price = 100000.0


##########
#初步转换成秒块
##########
for forex_type, full_time, bid_price, ask_price in reader:
	bid_price = float(bid_price)
	ask_price = float(ask_price)

	if bid_price > max_price:
		max_price = bid_price
	if ask_price > max_price:
		max_price = ask_price

	if bid_price < min_price:
		min_price = bid_price
	if ask_price < min_price:
		min_price = ask_price

	# print forex_type, full_time, bid_price, ask_price

	(YMD, TIME) = full_time.split()
	year = YMD[:4]
	month = YMD[4:6]
	date = YMD[6:]
	(hour, minute, full_second) = TIME.split(":")
	(second, m_second) = full_second.split(".")
	# print year, month, date, hour, minute, second, m_second

	# if pointer[0] == 0:
	# 	pointer[0] = year
	# 	pointer[1] = month
	# 	pointer[2] = date
	# 	pointer[3] = hour
	# 	pointer[4] = minute
	# 	pointer[5] = second

	while True:
		if pointer[0] == year and pointer[1] == month and pointer[2] == date and pointer[3] == hour and pointer[4] == minute and pointer[5] == second:
			bid_price_block.append(bid_price)
			ask_price_block.append(ask_price)
			m_second_block.append(float(m_second))
			length_counter += 1
			break
		else:
			sec_counter += 1

			if pointer[0] != 0:
				second_block_data.append([
						pointer[0], pointer[1], pointer[2], pointer[3], pointer[4], pointer[5], bid_price_block, ask_price_block, m_second_block
					])


			sum_length += length_counter
			# print length_counter, sum_length
			if length_counter < min_length and year != 0:
				min_length = length_counter

			length_counter = 0
			bid_price_block = []
			ask_price_block = []
			m_second_block = []
			pointer[0] = year
			pointer[1] = month
			pointer[2] = date
			pointer[3] = hour
			pointer[4] = minute
			pointer[5] = second


	row_counter += 1
	if row_counter%1000000 == 0:
		print "#",row_counter
		# break
	# break

print "MIN LENGTH",min_length
print "AVR LENGTH",float(sum_length)/float(sec_counter)
# print second_block_data[0]
# print second_block_data[1]
# print "OVER PRICE ",over_price_counter/2
print "ROW COUNTER",row_counter
print "MAX PRICE",max_price
print "MIN PRICE",min_price






##########
# 规范秒块
##########
normal_second_block_data = []

current_time = second_block_data[0][:6]

limit_time = second_block_data[-1][:6]
limit_datetime = datetime.datetime(int(limit_time[0]),int(limit_time[1]),int(limit_time[2]),int(limit_time[3]),int(limit_time[4]),int(limit_time[5]))

# print second_block_data[0]
# print second_block_data[1]
# print second_block_data[2]
# print second_block_data[3]
# print second_block_data[4]
# print second_block_data[5]
i = 0
while True:
	try:
		current_bid_price = second_block_data[i][6][0]
		current_ask_price = second_block_data[i][7][0]
		break
	except:
		i += 1
		continue

second_block_data_pointer = 0
second_block_data_pointer_limit = len(second_block_data)

second_counter = 0
second_blank_counter = 0

pre_bid_price = False
pre_ask_price = False


while True:
	has_sec_block = False
	if current_time[0] == second_block_data[second_block_data_pointer][0] and \
		current_time[1] == second_block_data[second_block_data_pointer][1] and \
		current_time[2] == second_block_data[second_block_data_pointer][2] and \
		current_time[3] == second_block_data[second_block_data_pointer][3] and \
		current_time[4] == second_block_data[second_block_data_pointer][4] and \
		current_time[5] == second_block_data[second_block_data_pointer][5]:
		has_sec_block = True
	if current_time[0] != second_block_data[second_block_data_pointer][0] or \
		current_time[1] != second_block_data[second_block_data_pointer][1] or \
		current_time[2] != second_block_data[second_block_data_pointer][2]:
		# 换天
		has_sec_block = True
		current_time[0] = second_block_data[second_block_data_pointer][0]
		current_time[1] = second_block_data[second_block_data_pointer][1]
		current_time[2] = second_block_data[second_block_data_pointer][2]
		current_time[3] = second_block_data[second_block_data_pointer][3]
		current_time[4] = second_block_data[second_block_data_pointer][4]
		current_time[5] = second_block_data[second_block_data_pointer][5]
	if has_sec_block:
		# 有现成的秒快
		second_block = " ".join(current_time).split()
		current_bid_price_block = []
		current_ask_price_block = []
		# 用来参考的值
		bid_price_list = second_block_data[second_block_data_pointer][6]
		ask_price_list = second_block_data[second_block_data_pointer][7]
		m_second_list = second_block_data[second_block_data_pointer][8]
		for x in range(SECOND_VOLUME):
			second_cuter_floor = 1000.0/float(SECOND_VOLUME)*float(x)
			second_cuter_roof = 1000.0/float(SECOND_VOLUME)*float(x+1)
			# 找到符合时间限制的最后价格
			m_second_pointer = -1
			for xx in range(len(m_second_list)):
				if m_second_list[xx] < second_cuter_roof and m_second_list[xx] > second_cuter_floor:
					m_second_pointer = xx
			if m_second_pointer < 0:
				# 没有
				pass
			else:
				# 有
				pre_bid_price = current_bid_price
				pre_ask_price = current_ask_price
				current_bid_price = float(bid_price_list[m_second_pointer])
				current_ask_price = float(ask_price_list[m_second_pointer])

			# # 转换成百分比数据
			# if pre_bid_price == False:
			# 	pre_bid_price = current_bid_price
			# 	pre_ask_price = current_ask_price

			# bid_price = (pre_bid_price/current_bid_price - 1)*PRICE_SCALE
			# ask_price = (pre_ask_price/current_ask_price - 1)*PRICE_SCALE

			# if bid_price > 1.0:
			# 	bid_price = 1.0
			# 	over_price_counter += 1
			# elif bid_price < -1.0:
			# 	bid_price = -1.0
			# 	over_price_counter += 1
			# if ask_price > 1.0:
			# 	ask_price = 1.0
			# 	over_price_counter += 1
			# elif ask_price < -1.0:
			# 	ask_price = -1.0
			# 	over_price_counter += 1
			bid_price = current_bid_price
			ask_price = current_ask_price

			# 添加
			current_bid_price_block.append(bid_price)
			current_ask_price_block.append(ask_price)

			# print current_bid_price,current_ask_price
		second_block += current_bid_price_block
		second_block += current_ask_price_block
		second_block_data_pointer += 1
		if second_block_data_pointer >= second_block_data_pointer_limit:
			break
	else:
		# 没有现成的秒快
		second_blank_counter += 1
		second_block = " ".join(current_time).split()
		second_block += [current_bid_price for x in range(SECOND_VOLUME)]
		# second_block += [0.0 for x in range(SECOND_VOLUME)] # 百分比
		second_block += [current_ask_price for x in range(SECOND_VOLUME)]
		# second_block += [0.0 for x in range(SECOND_VOLUME)] # 百分比
	# print second_block
	# 添加进
	normal_second_block_data.append(second_block)

	# 秒数+1
	current_time_datetime = datetime.datetime(int(current_time[0]),int(current_time[1]),int(current_time[2]),int(current_time[3]),int(current_time[4]),int(current_time[5]))
	current_time_datetime = current_time_datetime + datetime.timedelta(seconds=1)
	current_time[0] = current_time_datetime.strftime('%Y')
	current_time[1] = current_time_datetime.strftime('%m')
	current_time[2] = current_time_datetime.strftime('%d')
	current_time[3] = current_time_datetime.strftime('%H')
	current_time[4] = current_time_datetime.strftime('%M')
	current_time[5] = current_time_datetime.strftime('%S')

	if current_time_datetime > limit_datetime:
		break

	second_counter += 1
	if second_counter%86400 == 0:
		print current_time
		# break
	# break
print "SECOND COUNTER",second_counter
print "BLANK SEC COUNTER",second_blank_counter
print "BLANK PERCENT",float(second_blank_counter)/float(second_counter)



##########
# 组合输入输出，写入文件
##########
writer = csv.writer(open(output_path+output_file_name,"wb"))

HISTORY_SEC_LIMIT = SECOND_VOLUME*2*60
history_second_prices = []

# 转变成每日开盘价的百分比*10
open_day = [0,0,0]
open_day[0] = normal_second_block_data[0][0]
open_day[1] = normal_second_block_data[0][1]
open_day[2] = normal_second_block_data[0][2]
price_day_open = normal_second_block_data[0][6] # 开盘价用的是bid开盘价格
# price_day_open_ask = normal_second_block_data[0][6+SECOND_VOLUME]

s_0_01 = 0
s_0_05 = 0
s_0_1 = 0
s_0_5 = 0
s_1 = 0
s_all = 0

writer_counter = 0

history_changes = []

for second_block in normal_second_block_data:
	# 计算历史单位时间的改变
	history_second_prices += second_block[6:]

	if len(history_second_prices) > HISTORY_SEC_LIMIT:
		history_second_prices = history_second_prices[-HISTORY_SEC_LIMIT:]

	op_bid_price = history_second_prices[0]
	op_ask_price = history_second_prices[SECOND_VOLUME]
	for x in range(len(history_second_prices)/2/60):
		if x == 0:
			continue
		current_bid_growth = history_second_prices[x*(2*SECOND_VOLUME)]-op_bid_price
		current_ask_growth = history_second_prices[x*(2*SECOND_VOLUME)+SECOND_VOLUME]-op_bid_price
		history_changes.append(current_bid_growth*3125)
		history_changes.append(current_ask_growth*3125)

	# # 新日期
	# if second_block[0] != open_day[0] or second_block[1] != open_day[1] or second_block[2] != open_day[2]:
	# 	open_day[0] = second_block[0]
	# 	open_day[1] = second_block[1]
	# 	open_day[2] = second_block[2]
	# 	price_day_open = second_block[6]

	# # prices = [float('%0.4f'%((x/price_day_open-1.0)*12.5)) for x in second_block[6:]]
	# prices = []

	# for x in second_block[6:]:
	# 	v = float('%0.4f'%((x/price_day_open-1.0)*25000))
	# 	if v > 1.0:
	# 		v = 1.0
	# 	prices.append(v)

	# # 统计部分
	# s_all += 1
	# if prices[0] < 0.01:
	# 	s_0_01 += 1
	# elif prices[0] < 0.05:
	# 	s_0_05 += 1
	# elif prices[0] < 0.1:
	# 	s_0_1 += 1
	# elif prices[0] < 0.5:
	# 	s_0_5 += 1
	# elif prices[0] == 1.0:
	# 	s_1 += 1

	# # print prices
	# # break

	writer.writerow(second_block[6:])
	writer_counter += 1
	if writer_counter%100000 == 0:
		print "writer counter",writer_counter
print "writer counter",writer_counter

# print "<0.01", float(s_0_01)/float(s_all)
# print "<0.05", float(s_0_05)/float(s_all)
# print "<0.1 ", float(s_0_1)/float(s_all)
# print "<0.5 ", float(s_0_5)/float(s_all)
# print "<1   ", 1.0 - float(s_0_01+s_0_05+s_0_1+s_0_5+s_1)/float(s_all)
# print ">=1  ", float(s_1)/float(s_all)

print "AVR GROWTH", sum(history_changes)/float(len(history_changes))