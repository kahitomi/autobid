# -*- coding: utf-8 -*-
# 外汇s2s预测训练

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time, datetime, csv

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# from tensorflow.models.rnn.translate import data_utils
# from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.ops import variable_scope

# from common import tf_serving
# from tensorflow_serving.session_bundle import exporter


import matplotlib.pyplot as plt

import talib


# from common import config
if __name__ == "__main__":

	import seq2seq as seq2seq_model
	# from IAS import tf_seq2seq_one2one as seq2seq_model
	# from IAS import word2vec, word_segmentation
	import time_align, norm

	if len(sys.argv) < 2:
		raise ValueError("Please enter the action [train, test]")

	ACTION = sys.argv[1]

	if ACTION == "train":
		if len(sys.argv) < 3:
			raise ValueError("Please enter the source forex csv file name which should be in src/data/")

		CSV_NAME = sys.argv[2]
		SAVE_NAME = CSV_NAME.split(".")[0]

	if ACTION == "test":
		TEST_CSV_NAME = "NZDUSD-2016-06-day1-test.csv"
		if len(sys.argv) >= 4:
			TEST_CSV_NAME = sys.argv[3]

		if len(sys.argv) < 3:
			raise ValueError("Please enter the model name")
		else:
			SAVE_NAME = sys.argv[2]
else:
	from src.model import seq2seq as seq2seq_model
	from src.model import time_align, norm

	SAVE_NAME = "relase"


SOURCE_PATH = "src/data/forex/"




SECOND_VOLUME = 2*2 # values/second
DATA_DIS = 60
BASE_LENGTH = 60 # seconds


NUMBER_SPLIT = 100
BASIC_SPLIT = 0.00001

IFSAVE = True
IFTEST = False

VOLUME_differ = []
VOLUME = [99999999, 0]

COMPRESS = 2


sess_config = tf.ConfigProto()
# sess_config.gpu_options.allocator_type = 'BFC'
# sess_config.gpu_options.allow_growth = True



# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = (30, 2)
bucket = _buckets

tf.app.flags.DEFINE_float("export_version", 0.05, "Export version.")


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

# tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")

tf.app.flags.DEFINE_integer("size", 50, "Size of each model layer.")

tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
# tf.app.flags.DEFINE_integer("source_vocab_size", BASE_LENGTH*SECOND_VOLUME*NUMBER_SPLIT, "English vocabulary size.")
# tf.app.flags.DEFINE_integer("target_vocab_size", BASE_LENGTH*SECOND_VOLUME*NUMBER_SPLIT, "French vocabulary size.")
tf.app.flags.DEFINE_integer("source_vocab_size", 10, "English vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 10, "French vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "src/model/forex/"+SAVE_NAME, "Data directory")
tf.app.flags.DEFINE_string("train_dir", "src/model/forex/"+SAVE_NAME, "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")

# tf.app.flags.DEFINE_integer("steps_per_checkpoint", 8800, "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 3000, "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS













def read_data(source_path, max_size=None, test=None):
	"""Read data from source and target files and put into buckets.
	Args:
		source_path: path to the files with token-ids for the source language.
		target_path: path to the file with token-ids for the target language;
			it must be aligned with the source file: n-th line contains the desired
			output for n-th line from the source_path.
		max_size: maximum number of lines to read, all other will be ignored;
			if 0 or None, data files will be read completely (no limit).
	Returns:
		data_set: a list of length len(_buckets); data_set[n] contains a list of
			(source, target) pairs read from the provided data files that fit
			into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
			len(target) < _buckets[n][1]; source and target are lists of token-ids.
	"""

	# data_set = [[] for _ in _buckets]

	# file_obj = open(source_path)
	# reader = csv.reader(file_obj)

	counter = 0

	# file_list = os.listdir(source_path)
	# totle_counter = len(file_list)

 	
	# data_align_set = time_align.load(source_path)
	file_obj = open(source_path)
	reader = csv.reader(file_obj)
	data_align_set = [x for x in reader]


	# talib转化
	data_close = [ float(x[2]) for x in data_align_set ]
	data_high = [ float(x[3]) for x in data_align_set ]
	data_low = [ float(x[4]) for x in data_align_set ]

	# ax=plt.gca()


	data_EMA = talib.EMA(np.array(data_close), timeperiod=30)
	data_EMAs = talib.EMA(np.array(data_close), timeperiod=5)
	# plt.plot(data_EMA[:100])
	# plt.boxplot(np.array(data_EMAs))



	data_WILLR = talib.WILLR(np.array(data_high), np.array(data_low), np.array(data_close), timeperiod=14)
	data_WILLR = -data_WILLR/100.0 # normalize
	# plt.plot(data_WILLR[:100])
	# plt.boxplot(data_WILLR)


	data_RSI = talib.RSI(np.array(data_close), timeperiod=14)
	data_RSI = data_RSI/100.0 # normalize
	# plt.plot(data_RSI[:100])
	# plt.boxplot(np.array(data_RSI))


	data_slowk, data_slowd = talib.STOCH(np.array(data_high), np.array(data_low), np.array(data_close), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
	data_slowk = data_slowk/100.0 # normalize
	# plt.boxplot(np.array(data_slowk))
	data_slowd = data_slowd/100.0 # normalize
	# plt.boxplot(np.array(data_slowd))



	data_macd, data_macdsignal, data_macdhist = talib.MACD(np.array(data_close), fastperiod=12, slowperiod=26, signalperiod=9)
	# plt.plot(data_macd[:100])
	# ax.set_yticks(np.linspace(-0.0005,0.0005,11))

	data_macd = norm.tanh(data_macd, 0.0008)
	# plt.boxplot(data_macd)
	data_macdsignal = norm.tanh(data_macdsignal, 0.0008)
	# plt.boxplot(data_macdsignal)
	data_macdhist = norm.tanh(data_macdhist, 0.0003)
	# plt.boxplot(data_macdhist)


	data_upperband, data_middleband, data_lowerband = talib.BBANDS(np.array(data_close), timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)



	data_higher = np.array(data_high) - np.array(data_EMAs)
	data_higher = talib.EMA(data_higher, timeperiod=5)

	data_lower = np.array(data_low) - np.array(data_EMAs)
	data_lower = talib.EMA(data_lower, timeperiod=5)



	# plt.grid(True)
	# plt.show()




	data_set = []

	for i in range(50, len(data_close)-50):
		item = [
				# data_close[i],
				data_EMAs[i],
				data_EMA[i],
				data_WILLR[i],
				data_RSI[i],
				data_slowk[i],
				data_slowd[i],
				data_macd[i],
				data_macdsignal[i],
				data_macdhist[i],
				data_upperband[i],
				data_middleband[i],
				data_lowerband[i],
				data_higher[i],
				data_lower[i]
			]
		data_set.append(item)

		counter += 1
		###########
		# FOR TEST
		###########
		if IFTEST and counter > ((bucket[0]+bucket[1])*BASE_LENGTH/DATA_DIS+100):
			break








	# # 直接转入data
	# data_set = []
	# v_list = []
	# for item in data_align_set:
	# 	data_set.append(item)

	# 	v_list.append(int(item[-1]))
	# 	if len(v_list) > BASE_LENGTH/DATA_DIS:
	# 		v_list = v_list[(-int(BASE_LENGTH/DATA_DIS)):]

	# 	v = sum(v_list)
	# 	if v < VOLUME[0]:
	# 		VOLUME[0] = v
	# 	if v > VOLUME[1]:
	# 		VOLUME[1] = v

	# 	counter += 1

	# 	###########
	# 	# FOR TEST
	# 	###########
	# 	if IFTEST and counter > ((bucket[0]+bucket[1])*BASE_LENGTH/DATA_DIS+100):
	# 		break



	print ("===== Complete load data =====")
	print ("===== counter",counter,"=====")

	# print(VOLUME)

	return data_set






def create_model(session, forward_only, batch_size = FLAGS.batch_size, model_name = SAVE_NAME):
	"""Create translation model and initialize or load parameters in session."""
	model = seq2seq_model.Seq2SeqModel(
			FLAGS.source_vocab_size, FLAGS.target_vocab_size, _buckets,
			FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, batch_size,
			FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
			forward_only=forward_only)
	ckpt = tf.train.get_checkpoint_state("src/model/forex/"+model_name)
	if ckpt:
	# if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)

		if not forward_only:
			# set new learning rate
			print("Old Learning Rate: ",model.learning_rate.eval(session=session))
			new_learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)
			op = tf.assign(model.learning_rate, new_learning_rate)
			op_init = tf.initialize_variables([new_learning_rate])
			session.run([op_init])
			session.run([op])
			print("New Learning Rate: ",model.learning_rate.eval(session=session))

	else:
		print("Creating model with fresh parameters.")
		session.run(tf.initialize_all_variables())

	

	return model



def number_to_bools(n, base_n):
	v = float(n) - float(base_n)
	v = int(v/BASIC_SPLIT)
	v += int(NUMBER_SPLIT/2)
	if v >= NUMBER_SPLIT:
		v = NUMBER_SPLIT-1
	if v < 0:
		v = 0
	v_bools = [0.0 for x in range(NUMBER_SPLIT)]
	# print(v,"/",len(v_bools))
	v_bools[v] = 1.0

	return v_bools

def bools_to_number(bool_list):
	n = 0.0
	for x in range(len(bool_list)):
		if bool_list[x] == 1.0:
			n = float(x-int(NUMBER_SPLIT/2))*BASIC_SPLIT
			break
	return n


def output_to_number(output):
	number_list = []
	for x in range(BASE_LENGTH*SECOND_VOLUME):
		block = output[ (x*NUMBER_SPLIT) : ((x+1)*NUMBER_SPLIT) ]
		number_list.append( bools_to_number(block) )
	return number_list


differ_mm = []
def number_to_number(n, base_n):
	period = NUMBER_SPLIT*BASIC_SPLIT/2.0
	differ = float(n) - float(base_n)
	# differ_mm.append(float(n) - float(base_n))
	# differ += period

	# differ_mm.append(differ/(period*2.0))

	differ = differ/(period*2.0)

	# differ_mm.append((math.tanh(COMPRESS*differ)+1.0)/2.0)

	differ = (math.tanh(COMPRESS*differ)+1.0)/2.0

	# if differ < 0.0:
	# 	differ = 0.0
	# if differ > period*2.0:
	# 	differ = period*2.0
	# differ = differ/(period*2.0)


	return differ

def block_to_input(block, start_bid_price, start_ask_price):

	# # tick data
	# input_list = []
	# for b in block:
	# 	for n in b:
	# 		if n < SECOND_VOLUME/2:
	# 			input_list += number_to_bools(n,start_bid_price)
	# 		else:
	# 			input_list += number_to_bools(n,start_ask_price)

	# return input_list


	# low high open close data
	# open_bid = float( block[0][0] )
	# open_ask = float( block[0][int(SECOND_VOLUME/2)] )
	close_bid = number_to_number(float( block[-1][int(SECOND_VOLUME/2)-1] ), start_bid_price)
	close_ask = number_to_number(float( block[-1][-1] ), start_ask_price)
	low_bid = 1.0
	low_ask = 1.0
	high_bid = 0.0
	high_ask = 0.0
	for b in block:
		for n in b:
			if n < SECOND_VOLUME/2:
				# bid
				differ = number_to_number(n, start_bid_price)
				if differ < low_bid:
					low_bid = differ
				if differ > high_bid:
					high_bid = differ
			else:
				# ask
				differ = number_to_number(n, start_ask_price)
				if differ < low_ask:
					low_ask = differ
				if differ > high_ask:
					high_ask = differ

	# print(close_bid, low_bid, high_bid, close_ask, low_ask, high_ask)

	return (close_bid, low_bid, high_bid, close_ask, low_ask, high_ask)



def get_batch(data_set):
	data_set_size = len(data_set)
	# data random choose
	encoder_inputs = [ [] for x in range(bucket[0]) ]
	decoder_inputs = [ [] for x in range(bucket[1]) ]


	for x in range(FLAGS.batch_size):
		seed = random.randint(0, data_set_size-1-((bucket[0]+bucket[1])*BASE_LENGTH/DATA_DIS))

		l_index = int(seed+(bucket[0]-1)*BASE_LENGTH/DATA_DIS)
		r_index = int(seed+(bucket[0])*BASE_LENGTH/DATA_DIS)
		block = np.array(data_set[ l_index : r_index ])
		start_price_s = np.average(block[:,0])
		start_price = np.average(block[:,1])


		for bucket_id in range(sum(bucket)):
			l_index = int(seed+bucket_id*BASE_LENGTH/DATA_DIS)
			r_index = int(seed+(bucket_id+1)*BASE_LENGTH/DATA_DIS)
			block = np.array(data_set[ l_index : r_index ])

			_avr_emas = np.average(block[:,0])
			_avr_emas = number_to_number(_avr_emas, start_price_s)
			_avr_ema = np.average(block[:,1])
			_avr_ema = number_to_number(_avr_ema, start_price)

			_avr_wil = np.average(block[:,2])
			_avr_rsi = np.average(block[:,3])
			_avr_slowk = np.average(block[:,4])
			_avr_slowd = np.average(block[:,5])
			_avr_macd = np.average(block[:,6])
			_avr_macdsignal = np.average(block[:,7])
			_avr_macdhist = np.average(block[:,8])


			_avr_upperband = np.average(block[:,9])
			_avr_upperband = number_to_number(_avr_upperband, start_price_s)
			_avr_middleband = np.average(block[:,10])
			_avr_middleband = number_to_number(_avr_middleband, start_price_s)
			_avr_lowerband = np.average(block[:,11])
			_avr_lowerband = number_to_number(_avr_lowerband, start_price_s)


			_avr_higher = np.average(block[:,12])
			_avr_higher = number_to_number(_avr_higher, start_price_s)
			_avr_lower = np.average(block[:,13])
			_avr_lower = number_to_number(_avr_lower, start_price_s)



			_input = [_avr_emas, _avr_higher, _avr_lower, _avr_wil, _avr_rsi, _avr_slowk, _avr_slowd, _avr_macd, _avr_macdsignal, _avr_macdhist]
			# _input = [_avr_emas, _avr_ema, _avr_wil, _avr_rsi, _avr_slowk, _avr_slowd, _avr_macd, _avr_macdsignal, _avr_macdhist]

			if bucket_id < bucket[0]:
				encoder_inputs[bucket_id].append(_input)
			else:
				decoder_inputs[bucket_id-bucket[0]].append(_input)

				



	decoder_inputs = decoder_inputs[:-1]
	# print( np.array(decoder_inputs).shape )
	# add GO symble
	GO = [[[0.0 for x in range(FLAGS.target_vocab_size)] for x in range(FLAGS.batch_size)]]
	# print( np.array(GO).shape )
	decoder_inputs =  GO + decoder_inputs

	# print( np.array(encoder_inputs).shape )
	# print( np.array(decoder_inputs).shape )

	return (encoder_inputs, decoder_inputs)




def train(differ_mm=differ_mm, VOLUME_differ=VOLUME_differ):
	"""Train a en->fr translation model using WMT data."""
	# Prepare WMT data.

	# print("Preparing WMT data in %s" % FLAGS.data_dir)
	# en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
	# 		FLAGS.data_dir, FLAGS.source_vocab_size, FLAGS.target_vocab_size)

	with tf.Session() as sess:
		# Create model.
		print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
		model = create_model(sess, False)

		# Read data into buckets and compute their sizes.
		print ("Reading development and training data.")

		train_set = read_data(SOURCE_PATH+CSV_NAME)

		# return False


		# This is the training loop.
		step_time, loss = 0.0, 0.0
		accuracy = 0.0
		error = 0.0
		step_time_mini = 0.0
		current_step = 0
		previous_losses = []
		data_set_size = len(train_set)
		while True:
			# Get a batch and make a step.
			start_time = time.time()


			(encoder_inputs, decoder_inputs) = get_batch(train_set)

			#train
			stepout = model.step(sess, encoder_inputs, decoder_inputs, False)
			# _gn, step_loss, _ = model.step(sess, encoder_inputs_list, decoder_inputs_list, target_weights_list, bucket_id, False)

			# print("Gradient norm",stepout[0])
			# print("step_loss",stepout[1])
			# print ("-----STEP TIME",time.time()-start_time)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			step_time_mini += (time.time() - start_time) / 10.0
			loss += np.average(stepout[1]) / FLAGS.steps_per_checkpoint




			# Accuracy calculate
			p_out = np.array(stepout[2])
			t_out = np.array(decoder_inputs)

			p_out = p_out[:-1]
			t_out = t_out[1:]

			p_out = p_out[:,:,0:1]
			t_out = t_out[:,:,0:1]

			label_predict = ( np.array(p_out) >= 0.5 ).astype(int)
			label_target = ( np.array(t_out) >= 0.5 ).astype(int)

			results = np.equal(label_predict,label_target)
			# results = np.sum(results, axis=2)
			# results = (results == model.output_size).astype(int)
			
			# results = ( np.array(p_out)-np.array(t_out) < 2.0/NUMBER_SPLIT ).astype(int)


			true_accuracy = float(np.sum(results))/float(results.size)
			# true_accuracy = float(np.sum(results))/float((model.bucket[1]-1)*model.batch_size*model.output_size)

			accuracy += true_accuracy / FLAGS.steps_per_checkpoint

			_error = np.average(np.absolute(np.array(p_out) - np.array(t_out)))
			error += _error / FLAGS.steps_per_checkpoint
			current_step += 1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			
			# with tf.device("/job:localhost/task:0"):
			# /job:localhost/replica:0/task:0
			# if current_step % 10 == 0:
			# 	# print ("#########STEP TIME",step_time_mini)
			# 	step_time_mini = 0.0
			if current_step % FLAGS.steps_per_checkpoint == 0:	

				# print(p_out)
				# print(t_out)

				# print(np.array(p_out).shape)
				# print(np.array(t_out).shape)


				# Print statistics for the previous epoch.
				# perplexity = math.exp(loss) if loss < 300 else float('inf')
				perplexity = loss
				print (datetime.datetime.today())
				print (
					"==========",
					"global step", model.global_step.eval(session=sess),
					" learning rate", model.learning_rate.eval(session=sess),
					" step-time", step_time
					)
				print("LOSS",loss)
				print("ACCU",accuracy)
				print("ERRO",error)
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(FLAGS.train_dir, "forex.ckpt")
				if IFSAVE:
					model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				accuracy = 0.0
				error = 0.0
				# Run evals on development set and print their perplexity.
				# for bucket_id in xrange(len(_buckets)):
				# 	if len(dev_set[bucket_id]) == 0:
				# 		print("  eval: empty bucket %d" % (bucket_id))
				# 		continue
				# 	encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
				# 	_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
				# 	eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
				# 	print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
				# sys.stdout.flush()



				###########
				# 数据统计
				###########

				# differ_np = np.array(differ_mm)


				# plt.boxplot(differ_np)

				# # plt.hist(differ_np)

				# ax=plt.gca()
				# ax.set_yticks(np.linspace(-1.5,1.5,31))  
				# # ax.set_yticklabels( ('0.60', '0.65', '0.70', '0.75', '0.80','0.85','0.90','0.95'))




				# # differ_np = np.array(VOLUME_differ)

				# # plt.boxplot(differ_np)

				# # ax=plt.gca()
				# # ax.set_yticks(np.linspace(-1.5,1.5,31)) 



				# plt.grid(True)
				# plt.show()


				# # print(encoder_inputs[0])
				# # print(encoder_inputs[1])
				# # print(encoder_inputs[2])
				# # print(encoder_inputs[3])
				# # print(encoder_inputs[4])
				# # print(encoder_inputs[5])
				
				# # print(decoder_inputs[0])
				# # print(decoder_inputs[1])
				# break



				differ_mm = []
				VOLUME_differ = []




class decoder():

	# 初始化
	def __init__(self, model_name):

		sess = tf.InteractiveSession()
		self.model = create_model(sess, True, batch_size = 1, model_name = model_name)
		self.sess = sess

	# 预测
	def decode(self, data_EMAs, data_higher, data_lower, data_EMA, data_WILLR, data_RSI, data_slowk, data_slowd, data_macd, data_macdsignal, data_macdhist):
		if len(data_EMA) < bucket[0]+33:
			print("长度不符合")
			return False

		encoder_inputs = [ [] for x in range(bucket[0]) ]
		decoder_inputs = [ [] for x in range(bucket[1]) ]
		start_price_s = data_EMAs[-1]
		start_price = data_EMA[-1]
		base_point = -bucket[0]
		for bucket_id in range(bucket[0]):

			_avr_emas = number_to_number(data_EMAs[base_point+bucket_id], start_price_s)
			_avr_ema = number_to_number(data_EMA[base_point+bucket_id], start_price)
			_avr_wil = data_WILLR[base_point+bucket_id]
			_avr_rsi = data_RSI[base_point+bucket_id]
			_avr_slowk = data_slowk[base_point+bucket_id]
			_avr_slowd = data_slowd[base_point+bucket_id]
			_avr_macd = data_macd[base_point+bucket_id]
			_avr_macdsignal = data_macdsignal[base_point+bucket_id]
			_avr_macdhist = data_macdhist[base_point+bucket_id]
			_avr_higher = data_higher[base_point+bucket_id]
			_avr_lower = data_lower[base_point+bucket_id]


			_input = [_avr_emas, _avr_higher, _avr_lower, _avr_wil, _avr_rsi, _avr_slowk, _avr_slowd, _avr_macd, _avr_macdsignal, _avr_macdhist]
			# _input = [_avr_emas, _avr_ema, _avr_wil, _avr_rsi, _avr_slowk, _avr_slowd, _avr_macd, _avr_macdsignal, _avr_macdhist]

			encoder_inputs[bucket_id].append(_input)
		
		decoder_inputs[bucket_id-bucket[0]].append(_input)

		zeros = [[0.0 for x in range(FLAGS.target_vocab_size)] for x in range(1)]
		decoder_inputs =  [zeros for _ in range(bucket[1])]

		stepout = self.model.step(self.sess, encoder_inputs, decoder_inputs, True)

		return stepout[2][:-1]


	def close(self):
		self.sess.close()
		print ("CLOSED")

	


def self_test():
	"""Test the model."""
	with tf.Session() as sess:
		print("Test for Forex model.")
		# Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
		model = create_model(sess, True)

		model.batch_size = 10

		test_set = read_data(SOURCE_PATH+TEST_CSV_NAME)

		# 24 picese wheel
		data_size = len(test_set)
		wheel_part_number = 7
		wheel = [ (data_size/wheel_part_number)*(x+1) for x in range(wheel_part_number)]


		final_accu_collect = [ [] for x in range(wheel_part_number) ]
		final_error_collect = [ [] for x in range(wheel_part_number) ]
		final_error_dis_collect = [ [] for x in range(wheel_part_number) ]

		epo = 10.0*wheel_part_number
		epo_counter = 0
		while epo_counter <= epo:
			seed_wheel = random.randint(0, wheel_part_number-1)
			range_pair = [0, wheel[seed_wheel]]
			if seed_wheel != 0:
				range_pair[0] = wheel[seed_wheel-1]

			# print (range_pair)

			(encoder_inputs, decoder_inputs) = get_batch(test_set[int(range_pair[0]):int(range_pair[1])])
			# # data random choose
			# encoder_inputs = [ [] for x in range(bucket[0]) ]
			# decoder_inputs = [ [] for x in range(bucket[1]) ]
			# for x in range(model.batch_size):
			# 	# print(seed_wheel)
			# 	# print(wheel[seed_wheel] - data_size/wheel_part_number)
			# 	# print(wheel[seed_wheel]-1-(bucket[0]*BASE_LENGTH)-(bucket[1]*BASE_LENGTH))
			# 	seed = random.randint(
			# 			int(wheel[seed_wheel] - data_size/wheel_part_number),
			# 			int(wheel[seed_wheel]-1-(bucket[0]*BASE_LENGTH)-(bucket[1]*BASE_LENGTH))
			# 		)

			# 	start_bid_price = test_set[seed][0]
			# 	start_ask_price = test_set[seed][int(SECOND_VOLUME/2)]

			# 	for i in range(bucket[0]):
			# 		block = test_set[ (seed+i*BASE_LENGTH) : (seed+(i+1)*BASE_LENGTH)]
			# 		encoder_inputs[i].append( block_to_input(block, start_bid_price, start_ask_price) )
			# 	for i in range(bucket[1]):
			# 		block = test_set[ (seed+(bucket[0]+i)*BASE_LENGTH) : (seed+(bucket[0]+i+1)*BASE_LENGTH)]
			# 		decoder_inputs[i].append( block_to_input(block, start_bid_price, start_ask_price) )

			# decoder_inputs = decoder_inputs[:-1]
			# # add GO symble
			# decoder_inputs = [[[0.0 for x in range(model.input_size)] for x in range(model.batch_size)]] + decoder_inputs





			#test
			stepout = model.step(sess, encoder_inputs, decoder_inputs, True)

			p_out = np.array(stepout[2])
			t_out = np.array(decoder_inputs)

			p_out = p_out[:-1]
			t_out = t_out[1:]

			p_out = p_out[:,:, 0]
			t_out = t_out[:,:, 0]

			# # Error calculate
			# p_price = [ [ [] for y in range(model.batch_size) ] for x in range(bucket[1]-1)]
			# t_price = [ [ [] for y in range(model.batch_size) ] for x in range(bucket[1]-1)]
			# for step in range(bucket[1]-1):
			# 	for batch_number in range(model.batch_size):
			# 		p_price[step][batch_number] = output_to_number(p_out[step][batch_number])
			# 		t_price[step][batch_number] = output_to_number(t_out[step][batch_number])
			# # print(np.array(p_price).shape)
			# # print(np.array(t_price).shape)

			# error_price = np.absolute(np.array(p_price)-np.array(t_price))

			p_out_real = 2.0*np.array(p_out)-1.0
			t_out_real = 2.0*np.array(t_out)-1.0

			p_out_real[p_out_real > 0.999] = 0.999
			p_out_real[p_out_real < -0.999] = -0.999
			t_out_real[t_out_real > 0.999] = 0.999
			t_out_real[t_out_real < -0.999] = -0.999

			p_out_real = np.arctanh( p_out_real ) / COMPRESS
			t_out_real = np.arctanh( t_out_real ) / COMPRESS

			error_price = np.absolute(p_out_real-t_out_real)

			# error_price = np.absolute(np.array(p_out)-np.array(t_out))
			error_price = np.average(error_price, axis=1)
			# error_price = np.average(error_price, axis=1)

			# print(p_price[-1][0])

			# print(np.array(t_price).shape)

			final_error_collect[seed_wheel].append(error_price)



			error_dis = np.sqrt(np.power(p_out_real-t_out_real, 2))

			# error_dis = np.sqrt(np.power(np.array(p_out)-np.array(t_out), 2))
			error_dis = np.average(error_dis, axis=1)
			# error_dis = np.average(error_dis, axis=1)

			final_error_dis_collect[seed_wheel].append(error_dis)


			# start_bid_price = test_set[seed][0]
			# start_ask_price = test_set[seed][int(SECOND_VOLUME/2)]

			# output_to_number(output)


			
			# Accuracy calculate

			# label_predict = ( np.array(p_out) >= 0.5 ).astype(int)
			# label_target = np.array(t_out)



			label_predict = ( np.array(p_out) >= 0.5 ).astype(int)
			label_target = ( np.array(t_out) >= 0.5 ).astype(int)

			results = np.equal(label_predict,label_target)


			# results = (np.absolute(p_out_real-t_out_real) < 0.03).astype(int)


			# results = np.sum(results, axis=2)
			# results = (results == model.output_size).astype(int)

			# print (np.sum(results, axis=1)/float(model.batch_size))
			# print (results.shape)

			# true_accuracy = np.sum(results, axis=1)/float(model.batch_size)

			# true_accuracy = float(np.sum(results))/float((model.bucket[1]-1)*model.batch_size)


			true_accuracy = float(np.sum(results)) / float(results.size)
			final_accu_collect[seed_wheel].append(true_accuracy)
			# accuracy += true_accuracy / FLAGS.steps_per_checkpoint



			# break

			epo_counter+=1
			

		final_error = []
		for x in final_error_collect:
			if len(x) == 0:
				final_error.append([0.0 for _ in range(bucket[1]-1)])
			else:
				x = np.array(x)
				err = np.average(x, axis=0)
				final_error.append(err)
		final_error_dis = []
		for x in final_error_dis_collect:
			if len(x) == 0:
				final_error_dis.append([0.0 for _ in range(bucket[1]-1)])
			else:
				x = np.array(x)
				err = np.average(x, axis=0)
				final_error_dis.append(err)

		final_accu = []
		for x in final_accu_collect:
			if len(x) == 0:
				final_accu.append([0.0 for _ in range(bucket[1]-1)])
			else:
				x = np.array(x)
				accu = np.average(x, axis=0)
				final_accu.append(accu)

		# print ("=====FINAL UP DOWN=====",error)
		print ("=====SUB   ACCU=====")
		for x in range(len(final_accu)):
			print ("#",x,"#",final_accu[x])

		print ("=====FINAL ACCU=====")
		ACCU = [ np.average(np.array(x)) for x in final_accu]
		print (np.average(np.array(ACCU)))

		print ("=====SUB   ERRO=====")
		for x in range(len(final_error)):
			print ("#",x,"#",final_error[x])

		print ("=====FINAL ERRO=====")
		ERRO = [ np.average(np.array(x)) for x in final_error]
		print (np.average(np.array(ERRO)))

		# print ("=====SUB   DIST=====")
		# for x in range(len(final_error_dis)):
		# 	print ("#",x,"#",final_error_dis[x])

		# print ("=====FINAL DIST=====")
		# ERRO = [ np.average(np.array(x)) for x in final_error_dis]
		# print (np.average(np.array(ERRO)))


def model_test():
	"""Test the s2s model."""
	with tf.Session() as sess:
		print("Model-test for s2s model.")
	# Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
	model = seq2seq_model.Seq2SeqModel(input_size=5, output_size=5, buckets=[(3, 3)], size=200, num_layers=2, max_gradient_norm=5.0, batch_size=2, learning_rate=0.5, learning_rate_decay_factor=0.9, num_samples=8)

	sess.run(tf.initialize_all_variables())

	# Fake data set for both the (3, 3) bucket.
	data_set = [[]]
	for _ in range(50):
		batch_input = []
		batch_output = []
		for __ in range(model.buckets[0][0]):
			item = [0.0 for _ in range(model.input_size)]
			for _ in range(1):
				item[random.randint(0, model.input_size-1)] = 1.0
			batch_input.append(item)
		for __ in range(model.buckets[0][1]):
			item = [0.0 for _ in range(model.input_size)]
			for _ in range(1):
				item[random.randint(0, model.input_size-1)] = 1.0
			batch_output.append(item)
		one_batch = [batch_input, batch_output]
		data_set[0].append(one_batch)
	# data_set = [[
	# 		[
	# 			[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
	# 			[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
	# 		],
	# 		[
	# 			[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
	# 			[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
	# 		]
	# 	]]
	for _ in xrange(5001):  # Train the fake model for 5 steps.
		bucket_id = 0

		encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)

		# print( np.array(encoder_inputs).shape )
		# print( np.array(decoder_inputs).shape )
		# print( np.array(target_weights).shape )

		stepout = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

		if _ % 500 == 0:
			print("@@@@@@@@@@@@@@",_)
			p_out = np.array(stepout[2])
			t_out = np.array(decoder_inputs)
			# print(p_out[0][0][:5])
			# print(t_out[0][0][:5])
			error = np.average(np.absolute(t_out-p_out))
			print("LOSS",stepout[1])



			label_predict = ( np.array(p_out) >= 0.5 ).astype(int)
			label_target = np.array(t_out)
			print(np.array(label_predict).shape)
			print(np.array(label_target).shape)


			results = np.equal(label_predict,label_target)
			# print results
			results = np.sum(results, axis=2)
			print(results)
			results = (results == model.output_size).astype(int)
			# print results
			# print np.array(results).shape
			print("True Number",np.sum(results),"/", model.buckets[bucket_id][1]*model.batch_size)




def main(_):
	if FLAGS.self_test:
		self_test()
	elif FLAGS.decode:
		decode()
	else:
		train()


if __name__ == "__main__":

	# model_test()

	if ACTION == "train":
		train()
	elif ACTION == "test":
		self_test()



	# tf.app.run()
	# read_data(SOURCE_PATH+CSV_NAME)
	# train()
	# self_test()