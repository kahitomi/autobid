# -*- coding: utf-8 -*-
# 外汇s2s预测server
import sys

from flask import Flask, request, json, Response, redirect, url_for, render_template, jsonify

from gevent import monkey
from gevent.wsgi import WSGIServer



from src.model import forex_train


if len(sys.argv) < 3:
	raise ValueError("Please enter the MODEL_NAME and host PORT")

MODEL_NAME = sys.argv[1]
PORT = int(sys.argv[2])


predictor = forex_train.decoder(MODEL_NAME)


app = Flask(__name__)
app.debug = True





# 预测接口
@app.route('/predict', methods=['POST'])
def predict():

	prev_data = json.loads(request.form['json'])

	print prev_data

	data_EMAs = prev_data["data_EMAs"]
	data_higher = prev_data["data_higher"]
	data_lower = prev_data["data_lower"]
	data_EMA = prev_data["data_EMA"]
	data_WILLR = prev_data["data_WILLR"]
	data_RSI = prev_data["data_RSI"]
	data_slowk = prev_data["data_slowk"]
	data_slowd = prev_data["data_slowd"]
	data_macd = prev_data["data_macd"]
	data_macdsignal = prev_data["data_macdsignal"]
	data_macdhist = prev_data["data_macdhist"]

	predict_results = predictor.decode(data_EMAs, data_higher, data_lower, data_EMA, data_WILLR, data_RSI, data_slowk, data_slowd, data_macd, data_macdsignal, data_macdhist)
	
	return json.dumps(predict_results)






http_server = WSGIServer(('', PORT), app)
print "Start Flask Successfully"
http_server.serve_forever()