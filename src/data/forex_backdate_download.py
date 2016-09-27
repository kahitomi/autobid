# -*- coding: utf-8 -*-
# 导入下载的外汇csv文件，生成用于训练的数据文件

import sys, csv, time, requests




if len(sys.argv) < 4:
	raise ValueError("Please enter the forex name(such as EUR_USD), start time and end time(2016-10-09)")

FOREX_NAME = sys.argv[1]
START_TIME = sys.argv[2]
END_TIME = sys.argv[3]


granularity = "S5"
# Top of the minute alignment

# “S5” - 5 seconds
# “S10” - 10 seconds
# “S15” - 15 seconds
# “S30” - 30 seconds
# “M1” - 1 minute
# Top of the hour alignment

# “M2” - 2 minutes
# “M3” - 3 minutes
# “M4” - 4 minutes
# “M5” - 5 minutes
# “M10” - 10 minutes
# “M15” - 15 minutes
# “M30” - 30 minutes
# “H1” - 1 hour
# Start of day alignment (default 17:00, Timezone/New York)

# “H2” - 2 hours
# “H3” - 3 hours
# “H4” - 4 hours
# “H6” - 6 hours
# “H8” - 8 hours
# “H12” - 12 hours
# “D” - 1 Day
# Start of week alignment (default Friday)

# “W” - 1 Week
# Start of month alignment (First day of the month)

# “M” - 1 Month


output_path = "src/data/forex/"
output_file_name = "BACKDATE_"+FOREX_NAME+"_"+START_TIME+"_"+END_TIME+".csv"


def main():

	start_time = time.strptime(START_TIME, "%Y-%m-%d")
	end_time = time.strptime(END_TIME, "%Y-%m-%d")

	writer = csv.writer(open(output_path+output_file_name,"wb"))
	writer_counter = 0

	is_complete = False
	while not is_complete:
		payload = {
				"instrument": FOREX_NAME,
				"granularity": granularity,
				"start": time.strftime('%Y-%m-%dT%H:%M:%S.000000Z', start_time),
				# "end": END_TIME,
				"candleFormat": "midpoint", # midpoint bidask
				"includeFirst": "false",
				"count": 4000
			}
		response = requests.get("https://api-fxtrade.oanda.com/v1/candles", params=payload)
		json_response = response.json()

		if not json_response.has_key("candles"):
			print response.text

		print "====="
		print json_response["candles"][0]["time"]
		print json_response["candles"][-1]["time"]

		for item in json_response["candles"]:
			start_time = time.strptime(item["time"], "%Y-%m-%dT%H:%M:%S.000000Z")
			if start_time >= end_time:
				is_complete = True
				print "LAST"
				print item["time"]
				break
			# _out = [ 
			# 		item["time"],
			# 		item["openBid"],
			# 		item["closeBid"],
			# 		item["highBid"],
			# 		item["lowBid"],
			# 		item["openAsk"],
			# 		item["closeAsk"],
			# 		item["highAsk"],
			# 		item["lowAsk"],
			# 		item["volume"],
			# 	]
			_out = [ 
					item["time"],
					item["openMid"],
					item["closeMid"],
					item["highMid"],
					item["lowMid"],
					item["volume"]
				]
			writer.writerow(_out)
			writer_counter += 1

	print "====="
	print "Raw", writer_counter
	print "Complete save",output_file_name




if __name__ == "__main__":
	main()