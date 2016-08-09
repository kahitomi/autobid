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

import tf_seq2seq as seq2seq_model
# from IAS import tf_seq2seq_one2one as seq2seq_model
# from IAS import word2vec, word_segmentation

# from common import config

if len(sys.argv) < 2:
	raise ValueError("Please enter the source forex csv file name which should be in src/data/")

CSV_NAME = sys.argv[1]
TEST_CSV_NAME = "NZDUSD-2016-06.csv"

SOURCE_PATH = "src/data/forex/"

if len(sys.argv) < 3:
	SAVE_NAME = CSV_NAME.split(".")[0]
else:
	SAVE_NAME = sys.argv[2]


SECOND_VOLUME = 6 # values/second
BASE_LENGTH = 60 # seconds


sess_config = tf.ConfigProto()
# sess_config.gpu_options.allocator_type = 'BFC'
# sess_config.gpu_options.allow_growth = True



# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(15, 5)]


tf.app.flags.DEFINE_float("export_version", 0.05, "Export version.")


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")

tf.app.flags.DEFINE_integer("size", BASE_LENGTH*SECOND_VOLUME, "Size of each model layer.")

tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_vocab_size", BASE_LENGTH*SECOND_VOLUME, "English vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", BASE_LENGTH*SECOND_VOLUME, "French vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "src/model/forex/"+SAVE_NAME, "Data directory")
tf.app.flags.DEFINE_string("train_dir", "src/model/forex/"+SAVE_NAME, "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 18000, "How many training steps to do per checkpoint.")

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
	for second_prices in reader:
		data_set[0].append(second_prices)
		counter += 1

		###########
		# FOR TEST
		###########
		# if counter > 1400:
		# 	break


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




	print ("===== Complete load data =====")
	print ("===== counter",counter,"=====")
	return data_set






def create_model(session, forward_only):
	"""Create translation model and initialize or load parameters in session."""
	model = seq2seq_model.Seq2SeqModel(
			FLAGS.source_vocab_size, FLAGS.target_vocab_size, _buckets,
			FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
			FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
			num_samples = 512,
			forward_only=forward_only)
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt:
	# if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)

		# if not forward_only:
		# 	# set new learning rate
		# 	new_learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)
		# 	op = tf.assign(model.learning_rate, new_learning_rate)
		# 	op_init = tf.initialize_variables([new_learning_rate])
		# 	session.run([op_init,op])
		# 	print("New Learning Rate: ",model.learning_rate.eval(session=session))

	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())

	

	return model


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

		# checkpoint_path = os.path.join(FLAGS.train_dir, "mindflow.ckpt")
		# model.saver.save(sess, checkpoint_path, global_step=model.global_step)

		# Read data into buckets and compute their sizes.
		print ("Reading development and training data (limit: %d)."
					 % FLAGS.max_train_data_size)

		# # Set logs writer into folder /tmp/tensorflow_logs
		# summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph_def=sess.graph_def)

		train_set = read_data(SOURCE_PATH+CSV_NAME)

		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		train_total_size = float(sum(train_bucket_sizes))

		# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
		# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
		# the size if i-th training bucket, as used later.
		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
													 for i in xrange(len(train_bucket_sizes))]

		# This is the training loop.
		step_time, loss = 0.0, 0.0
		step_time_mini = 0.0
		current_step = 0
		previous_losses = []
		while True:
			# print("current_step",current_step)
			# Choose a bucket according to data distribution. We pick a random number
			# in [0, 1] and use the corresponding interval in train_buckets_scale.
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale))
											 if train_buckets_scale[i] > random_number_01])

			# Get a batch and make a step.
			start_time = time.time()
			# print("bucket_id",bucket_id)

			# get train data
			# encoder_inputs_list = []
			# decoder_inputs_list = []
			# target_weights_list = []
			# for task_num in range(TASK_NUM):
			get_start = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch_seq(train_set, bucket_id, BASE_LENGTH)
			# print ("get batch time",time.time()-get_start)

			#train
			stepout = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
			# _gn, step_loss, _ = model.step(sess, encoder_inputs_list, decoder_inputs_list, target_weights_list, bucket_id, False)

			# print("Gradient norm",stepout[0])
			# print("step_loss",stepout[1])
			# print ("-----STEP TIME",time.time()-start_time)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			step_time_mini += (time.time() - start_time) / 10.0
			loss += stepout[1] / FLAGS.steps_per_checkpoint
			current_step += 1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			
			# with tf.device("/job:localhost/task:0"):
			# /job:localhost/replica:0/task:0
			if current_step % 10 == 0:
				# print ("#########STEP TIME",step_time_mini)
				step_time_mini = 0.0
			if current_step % FLAGS.steps_per_checkpoint == 0:
				# Print statistics for the previous epoch.
				# perplexity = math.exp(loss) if loss < 300 else float('inf')
				perplexity = loss
				print (datetime.datetime.today())
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
							 "%.10f" % (model.global_step.eval(session=sess), model.learning_rate.eval(session=sess),
												 step_time, perplexity))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(FLAGS.train_dir, "mindflow.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
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
	"""Test the translation model."""
	with tf.Session() as sess:
		print("Self-test for Seq2seq model.")
		# Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
		model = create_model(sess, True)

		model.batch_size = 100

		test_set = read_data(SOURCE_PATH+CSV_NAME)

		test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
		test_total_size = float(sum(test_bucket_sizes))
		test_buckets_scale = [sum(test_bucket_sizes[:i + 1]) / test_total_size
													 for i in xrange(len(test_bucket_sizes))]
		bucket_id = 0
		i = 0
		loss = 0.0
		epo = 10.0
		error = 0.0
		while i <= epo:
			encoder_inputs, decoder_inputs, target_weights = model.get_batch_seq(test_set, bucket_id, BASE_LENGTH)
			#test
			stepout = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
			# _gn, step_loss, _ = model.step(sess, encoder_inputs_list, decoder_inputs_list, target_weights_list, bucket_id, False)

			print("Gradient norm",stepout[0])
			print("step_loss",stepout[1])
			# print ("-----STEP TIME",time.time()-start_time)

			# print (len(stepout[2]))
			# print (len(stepout[2][0]))
			# print (len(stepout[2][0][0]))

			# print (stepout[2][0][0])
			# print (decoder_inputs[0][0])

			predict_result = np.array(stepout[2])
			target_result = np.array(decoder_inputs)

			# print (predict_result.shape)
			# print (target_result.shape)

			error_result = target_result-predict_result

			error_result = np.average(error_result, axis=0)
			error_result = np.average(error_result, axis=1)
			_error = np.average(error_result,)

			print (error_result)
			print (_error)
			# print (error_result.shape)

			error += _error/epo

			break

			i+=1
			continue




			# predict_result = []
			# target_result = []
			# for x in range(model.batch_size):
			# 	p_value = 0.0
			# 	t_value = 0.0

			# 	for y in range(_buckets[0][1]):
			# 		for v in stepout[2][y][x]:
			# 			if v == 0.0:
			# 				continue
			# 			if p_value == 0.0:
			# 				p_value = v
			# 				continue
			# 			p_value *= v
			# 			# print(p_value)
			# 			# if p_value == float('Inf'):
			# 			# 	break
			# 		for v in decoder_inputs[y][x]:
			# 			if v == 0.0:
			# 				continue
			# 			if t_value == 0.0:
			# 				t_value = v
			# 				continue
			# 			t_value *= v

			# 	predict_result.append(p_value)
			# 	target_result.append(t_value)

			# print (predict_result)
			# print (target_result)
			# error_list = [target_result[x]-predict_result[x] for x in range(model.batch_size)]
			# _error = sum(error_list)/len(error_list)/epo
			# if _error == float('Inf'):
			# 	error += 100.0/epo
			# 	continue
			# error += _error
		print ("=====FINAL   ERROR=====",error)
		# print ("=====FINAL UP DOWN=====",error)








def main(_):
	if FLAGS.self_test:
		self_test()
	elif FLAGS.decode:
		decode()
	else:
		train()


if __name__ == "__main__":
	# tf.app.run()
	# read_data(SOURCE_PATH+CSV_NAME)
	train()
	# self_test()