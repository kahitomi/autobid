# -*- coding: utf-8 -*-
# 导入下载的外汇csv文件，生成用于训练的数据文件

import sys, csv, time, requests




if len(sys.argv) < 4:
	raise ValueError("Please enter the forex name(such as EUR_USD), start time and end time(2016-10-09)")

FOREX_NAME = sys.argv[1]
START_TIME = sys.argv[2]
END_TIME = sys.argv[3]


granularity = "S5"
output_path = "src/data/forex/"
output_file_name = FOREX_NAME+"_"+START_TIME+"_"+END_TIME+".csv"


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
				"candleFormat": "bidask",
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
			_out = [ 
					item["time"],
					item["openBid"],
					item["closeBid"],
					item["highBid"],
					item["lowBid"],
					item["openAsk"],
					item["closeAsk"],
					item["highAsk"],
					item["lowAsk"],
					item["volume"],
				]
			writer.writerow(_out)
			writer_counter += 1

	print "====="
	print "Raw", writer_counter
	print "Complete save",output_file_name




if __name__ == "__main__":
	main()