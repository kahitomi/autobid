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

import seq2seq as seq2seq_model
# from IAS import tf_seq2seq_one2one as seq2seq_model
# from IAS import word2vec, word_segmentation

# from common import config

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


SOURCE_PATH = "src/data/forex/"




SECOND_VOLUME = 2*2 # values/second
BASE_LENGTH = 30 # seconds

NUMBER_SPLIT = 50
BASIC_SPLIT = 0.00001

IFSAVE = False
IFTEST = True


sess_config = tf.ConfigProto()
# sess_config.gpu_options.allocator_type = 'BFC'
# sess_config.gpu_options.allow_growth = True



# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = (24, 6)
bucket = _buckets

tf.app.flags.DEFINE_float("export_version", 0.05, "Export version.")


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

# tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")

tf.app.flags.DEFINE_integer("size", 200, "Size of each model layer.")

tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
# tf.app.flags.DEFINE_integer("source_vocab_size", BASE_LENGTH*SECOND_VOLUME*NUMBER_SPLIT, "English vocabulary size.")
# tf.app.flags.DEFINE_integer("target_vocab_size", BASE_LENGTH*SECOND_VOLUME*NUMBER_SPLIT, "French vocabulary size.")
tf.app.flags.DEFINE_integer("source_vocab_size", 6, "English vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 6, "French vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "src/model/forex/"+SAVE_NAME, "Data directory")
tf.app.flags.DEFINE_string("train_dir", "src/model/forex/"+SAVE_NAME, "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")

# tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 800, "How many training steps to do per checkpoint.")

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

	data_set = [[] for _ in _buckets]

	file_obj = open(source_path)
	reader = csv.reader(file_obj)

	counter = 0

	# file_list = os.listdir(source_path)
	# totle_counter = len(file_list)

 

	# 直接转入data
	data_set = []
	for second_prices in reader:
		data_set.append(second_prices)
		counter += 1

		###########
		# FOR TEST
		###########
		if IFTEST and counter > ((bucket[0]+bucket[1])*BASE_LENGTH+8000):
			break




	# # 变换到基本长度
	# print("Start to Transform to basic units")
	# base_units = []
	# history_second_prices = []
	# for second_prices in reader:
	# 	if len(history_second_prices) == BASE_LENGTH:
	# 		_unit = []
	# 		for x in history_second_prices:
	# 			_unit += x
	# 		# print(_unit)
	# 		base_units.append(_unit)
	# 		# break
	# 		counter += 1
	# 		history_second_prices = []
	# 		if counter%10000 == 0:
	# 			print("Reading basic units",counter)
	# 			# break

	# 	history_second_prices.append(second_prices)
	# 	if len(history_second_prices) > BASE_LENGTH:
	# 		history_second_prices = history_second_prices[-BASE_LENGTH:]
	# file_obj.close()
	# print("Complete basic units",counter)

	# # 组合输入输出
	# print("Creating sources and targets")
	# counter = 0
	# source_unit_length = _buckets[0][0]
	# target_unit_length = _buckets[0][1]
	# max_unit_number = len(base_units)
	# for point in range(max_unit_number):
	# 	if point < source_unit_length or point > max_unit_number-target_unit_length:
	# 		continue
	# 	source_units = base_units[(point-source_unit_length):point]
	# 	target_units = base_units[point:(point+target_unit_length)]

	# 	data_set[0].append([source_units, target_units])

	# 	counter += 1

	# 	###########
	# 	# FOR TEST
	# 	###########
	# 	if counter > 200:
	# 		break




	print ("===== Complete load data =====")
	print ("===== counter",counter,"=====")
	return data_set






def create_model(session, forward_only):
	"""Create translation model and initialize or load parameters in session."""
	model = seq2seq_model.Seq2SeqModel(
			FLAGS.source_vocab_size, FLAGS.target_vocab_size, _buckets,
			FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
			FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
			forward_only=forward_only)
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
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


def number_to_number(n, base_n):
	period = NUMBER_SPLIT*BASIC_SPLIT/2.0
	differ = float(n) - float(base_n)
	differ += period

	differ = differ/(period*2.0)
	differ = (math.tanh(4.5*differ)+1.0)/2.0

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





def train():
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


			# data random choose
			encoder_inputs = [ [] for x in range(bucket[0]) ]
			decoder_inputs = [ [] for x in range(bucket[1]) ]
			for x in range(model.batch_size):
				seed = random.randint(0, data_set_size-1-(bucket[0]*BASE_LENGTH)-(bucket[1]*BASE_LENGTH))

				start_bid_price = train_set[seed][0]
				start_ask_price = train_set[seed][int(SECOND_VOLUME/2)]

				for i in range(bucket[0]):
					block = train_set[ (seed+i*BASE_LENGTH) : (seed+(i+1)*BASE_LENGTH)]
					encoder_inputs[i].append( block_to_input(block, start_bid_price, start_ask_price) )
				for i in range(bucket[1]):
					block = train_set[ (seed+(bucket[0]+i)*BASE_LENGTH) : (seed+(bucket[0]+i+1)*BASE_LENGTH)]
					decoder_inputs[i].append( block_to_input(block, start_bid_price, start_ask_price) )

			decoder_inputs = decoder_inputs[:-1]
			# add GO symble
			decoder_inputs = [[[0.0 for x in range(model.input_size)] for x in range(model.batch_size)]] + decoder_inputs

			# print( np.array(encoder_inputs).shape )
			# print( np.array(decoder_inputs).shape )

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

			label_predict = ( np.array(p_out) >= 0.5 ).astype(int)
			label_target = np.array(t_out)

			results = np.equal(label_predict,label_target)
			results = np.sum(results, axis=2)
			results = (results == model.output_size).astype(int)

			true_accuracy = float(np.sum(results))/float((model.bucket[1]-1)*model.batch_size)

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
				sys.stdout.flush()

			# break



def s2s(model, id_list):
	id_list.append(EOS)

	encoder_input = tf.placeholder(tf.int32, shape=[None],name="encoder_input")
	decoder_input = tf.placeholder(tf.int32, shape=[None],name="decoder_input")
	decode_input = seq2seq_model.data_utils["GO_ID"]

	with variable_scope.variable_scope("embedding_attention_seq2seq"):
		# Encoder.
		encoder_cell = tf.get_variable("encoder_cell")
		print(encoder_cell)

		# encoder_cell = rnn_cell.EmbeddingWrapper(
		# 	cell, embedding_classes=num_encoder_symbols,
		# 	embedding_size=embedding_size)
		# encoder_outputs, encoder_state = rnn.rnn(
		# 	encoder_cell, encoder_inputs, dtype=dtype)


		# for x in reversed(id_list):
			

		# 	# First calculate a concatenation of encoder outputs to put attention on.
		# 	top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
		# 	          for e in encoder_outputs]
		# 	attention_states = array_ops.concat(1, top_states)


		# 	break


def decode(questions):
	# with tfec2.TFEc2() as sess:
	with tf.Session("grpc://"+gpu_address+":2222") as sess:
	# with tf.Session() as sess:
		# Create model and load parameters.
		model = create_model(sess, True)
		model.batch_size = 1  # We decode one sentence at a time.

		# # Load vocabularies.
		# en_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.en" % FLAGS.source_vocab_size)
		# fr_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.fr" % FLAGS.target_vocab_size)
		# en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
		# _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

		# Decode from standard input.
		sys.stdout.write("> ")
		sys.stdout.flush()
		# sentence = sys.stdin.readline()

		# 瞎写
		for question in questions:
			print("++++++++++++++++++++")
			# print (question)
			words_list = word_segmentation.word_cuter(question).split()
			print (word_segmentation.word_cuter(question))
			id_list = []
			for x in words_list:
				try:
					id_list.append(word2vec.my_dictionary.token2id[x.decode("utf-8")])
				except:
					id_list.append(len(word2vec.my_dictionary.token2id)+999)
			# print (id_list)

			# s2s(model, id_list)

			token_ids = np.array(id_list)

			# print (token_ids)

			# # Get token-ids for the input sentence.
			# token_ids = data_utils.sentence_to_token_ids(sentence, en_vocab)
			# Which bucket does it belong to?
			bucket_id = min([b for b in xrange(len(_buckets))
											 if _buckets[b][0] > len(token_ids)])
			# Get a 1-element batch to feed the sentence to the model.

			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
					{bucket_id: [(token_ids, np.array([]))]}, bucket_id, sess)
			# Get output logits for the sentence.
			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

			# print (output_logits)
			# print (len(output_logits[0]))

			# # This is a greedy decoder - outputs are just argmaxes of output_logits.
			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
			# outputs = [int(np.argmax(logit, axis=0)) for logit in output_logits]
			# print (outputs)

			output_words = ""
			for x in outputs:
				if x == EOS:
					output_words += "<EOS>"
				elif x == word2vec.data_utils["EOT"]:
					output_words += "<EOT>"
				elif x == word2vec.data_utils["GO_ID"]:
					output_words += "<GO>"
				elif x == word2vec.data_utils["PAD_ID"]:
					output_words += "<PAD>"
				else:
					output_words += word2vec.my_dictionary.get(int(x),"<unk>")
			print (output_words)

			# # If there is an EOS symbol in outputs, cut them at that point.
			# if data_utils.EOS_ID in outputs:
			# 	outputs = outputs[:outputs.index(data_utils.EOS_ID)]
			# # Print out French sentence corresponding to outputs.
			# print(" ".join([rev_fr_vocab[output] for output in outputs]))
			# print("> ", end="")
			# sys.stdout.flush()
			
			# break

		# test_set = read_data(config.source_path+"/IAS/source/dialog/", FLAGS.max_train_data_size, test=True)

		# test_set = test_set[:10]
		# while sentence in test_set:
		# 	# Get token-ids for the input sentence.
		# 	token_ids = sentence
		# 	# Which bucket does it belong to?
		# 	bucket_id = min([b for b in xrange(len(_buckets))
		# 									 if _buckets[b][0] > len(token_ids)])
		# 	# Get a 1-element batch to feed the sentence to the model.
		# 	encoder_inputs, decoder_inputs, target_weights = model.get_batch(
		# 			{bucket_id: [(token_ids, [])]}, bucket_id)
		# 	# Get output logits for the sentence.
		# 	_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
		# 	# This is a greedy decoder - outputs are just argmaxes of output_logits.
		# 	outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
		# 	# If there is an EOS symbol in outputs, cut them at that point.
		# 	if data_utils.EOS_ID in outputs:
		# 		outputs = outputs[:outputs.index(data_utils.EOS_ID)]
		# 	# Print out French sentence corresponding to outputs.
		# 	print(" ".join([rev_fr_vocab[output] for output in outputs]))
		# 	print("> ", end="")
		# 	sys.stdout.flush()
		# 	sentence = sys.stdin.readline()


		# from tensorflow_serving.session_bundle import exporter
		# print ('Exporting trained model to', export_path)
		# saver = model.saver
		# model_exporter = exporter.Exporter(saver)
		# serving_inputs = {}
		# for x in model.encoder_inputs:
		# 	serving_inputs[x.name] = x
		# # serving_decode_tensor = {}
		# for x in model.decoder_inputs:
		# 	serving_inputs[x.name] = x
		# # serving_output_tensor = {}
		# for j in model.outputs:
		# 	for x in j:
		# 		serving_inputs[x.name] = x
		# print("create signature")
		# signature = exporter.generic_signature(serving_inputs)
		# print("exporter init")
		# model_exporter.init(sess.graph.as_graph_def(),
		#                     default_graph_signature=signature)
		# print("export model")s
		# model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)


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
		wheel_part_number = 24
		wheel = [ (data_size/wheel_part_number)*(x+1) for x in range(wheel_part_number)]


		final_accu_collect = [ [] for x in range(wheel_part_number) ]
		final_error_collect = [ [] for x in range(wheel_part_number) ]
		final_error_dis_collect = [ [] for x in range(wheel_part_number) ]

		epo = 100.0*wheel_part_number
		epo_counter = 0
		while epo_counter <= epo:
			seed_wheel = random.randint(0, wheel_part_number-1)
			# data random choose
			encoder_inputs = [ [] for x in range(bucket[0]) ]
			decoder_inputs = [ [] for x in range(bucket[1]) ]
			for x in range(model.batch_size):
				# print(seed_wheel)
				# print(wheel[seed_wheel] - data_size/wheel_part_number)
				# print(wheel[seed_wheel]-1-(bucket[0]*BASE_LENGTH)-(bucket[1]*BASE_LENGTH))
				seed = random.randint(
						int(wheel[seed_wheel] - data_size/wheel_part_number),
						int(wheel[seed_wheel]-1-(bucket[0]*BASE_LENGTH)-(bucket[1]*BASE_LENGTH))
					)

				start_bid_price = test_set[seed][0]
				start_ask_price = test_set[seed][int(SECOND_VOLUME/2)]

				for i in range(bucket[0]):
					block = test_set[ (seed+i*BASE_LENGTH) : (seed+(i+1)*BASE_LENGTH)]
					encoder_inputs[i].append( block_to_input(block, start_bid_price, start_ask_price) )
				for i in range(bucket[1]):
					block = test_set[ (seed+(bucket[0]+i)*BASE_LENGTH) : (seed+(bucket[0]+i+1)*BASE_LENGTH)]
					decoder_inputs[i].append( block_to_input(block, start_bid_price, start_ask_price) )

			decoder_inputs = decoder_inputs[:-1]
			# add GO symble
			decoder_inputs = [[[0.0 for x in range(model.input_size)] for x in range(model.batch_size)]] + decoder_inputs





			#test
			stepout = model.step(sess, encoder_inputs, decoder_inputs, True)

			p_out = np.array(stepout[2])
			t_out = np.array(decoder_inputs)

			p_out = p_out[:-1]
			t_out = t_out[1:]

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

			p_out_real = np.arctanh( p_out_real ) / 4.5
			t_out_real = np.arctanh( t_out_real ) / 4.5

			error_price = np.absolute(p_out_real-t_out_real)

			# error_price = np.absolute(np.array(p_out)-np.array(t_out))
			error_price = np.average(error_price, axis=1)
			error_price = np.average(error_price, axis=1)

			# print(p_price[-1][0])

			# print(np.array(t_price).shape)

			final_error_collect[seed_wheel].append(error_price)



			error_dis = np.sqrt(np.power(p_out_real-t_out_real, 2))

			# error_dis = np.sqrt(np.power(np.array(p_out)-np.array(t_out), 2))
			error_dis = np.average(error_dis, axis=1)
			error_dis = np.average(error_dis, axis=1)

			final_error_dis_collect[seed_wheel].append(error_dis)


			# start_bid_price = test_set[seed][0]
			# start_ask_price = test_set[seed][int(SECOND_VOLUME/2)]

			# output_to_number(output)


			
			# Accuracy calculate

			# label_predict = ( np.array(p_out) >= 0.5 ).astype(int)
			# label_target = np.array(t_out)

			# results = np.equal(label_predict,label_target)


			results = (np.absolute(p_out_real-t_out_real) < 0.1).astype(int)


			# results = np.sum(results, axis=2)
			# results = (results == model.output_size).astype(int)

			# print (np.sum(results, axis=1)/float(model.batch_size))
			# print (results.shape)

			true_accuracy = np.sum(results, axis=1)/float(model.batch_size)

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

		print ("=====SUB   DIST=====")
		for x in range(len(final_error_dis)):
			print ("#",x,"#",final_error_dis[x])

		print ("=====FINAL DIST=====")
		ERRO = [ np.average(np.array(x)) for x in final_error_dis]
		print (np.average(np.array(ERRO)))


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