# -*- coding: utf-8 -*-
# Tensorflow Seq2Seq模型

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
# from tensorflow.python.util import nest


class Seq2SeqModel(object):
	"""
	Sequence-to-sequence model with attention and for multiple buckets.
	This class implements a multi-layer recurrent neural network as encoder,
	and an attention-based decoder. This is the same as the model described in
	this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
	or into the seq2seq library for complete model implementation.
	This class also allows to use GRU cells in addition to LSTM cells, and
	sampled softmax to handle large output vocabulary size. A single-layer
	version of this model, but with bi-directional encoder, was presented in
		http://arxiv.org/abs/1409.0473
	and sampled softmax is described in Section 3 of the following paper.
		http://arxiv.org/abs/1412.2007
	"""

	def __init__(self, input_size, output_size, bucket, size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, use_lstm=False, forward_only=False):

		"""Create the model.
		Args:
			input_size: size of the source vocabulary.
			output_size: size of the target vocabulary.
			bucket: (I, O), where I specifies maximum input length
				that will be processed in that bucket, and O specifies maximum output
				length. Training instances that have inputs longer than I or outputs
				longer than O will be pushed to the next bucket and padded accordingly.
				We assume that the list is sorted, e.g., (2, 4).
			size: number of units in each layer of the model.
			num_layers: number of layers in the model.
			max_gradient_norm: gradients will be clipped to maximally this norm.
			batch_size: the size of the batches used during training;
				the model construction is independent of batch_size, so it can be
				changed after initialization if this is convenient, e.g., for decoding.
			learning_rate: learning rate to start with.
			learning_rate_decay_factor: decay learning rate by this much when needed.
			use_lstm: if true, we use LSTM cells instead of GRU cells.
			forward_only: if set, we do not construct the backward pass in the model.
		"""
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = size
		self.bucket = bucket
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
				self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		

		# Create the internal multi-layer cell for our RNN.
		single_cell = tf.nn.rnn_cell.GRUCell(size)
		# single_cell = tf.nn.rnn_cell.GRUCell(size, activation=tf.nn.relu)
		if use_lstm:
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
		cell = single_cell
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

		softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.output_size])
		softmax_b = tf.get_variable("softmax_b", [self.output_size])


		# The seq2seq function
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			# return attention_seq2seq(encoder_inputs, decoder_inputs, cell,  self.input_size, self.hidden_size, self.output_size, feed_previous=do_decode)
			# return basic_seq2seq(encoder_inputs, decoder_inputs, cell, self.input_size, self.hidden_size, self.output_size)

			with variable_scope.variable_scope("my_seq2seq"):

				wrapper_cell = tf.nn.rnn_cell.InputProjectionWrapper(cell, self.hidden_size, self.input_size)

				encoder_outputs, enc_state = rnn.rnn(wrapper_cell, encoder_inputs, dtype=dtypes.float32)


				if do_decode:
					def simple_loop_function(prev, _):
						_next = tf.greater_equal(prev, 0.5)
						_next = tf.to_float(_next)
						return _next	
					loop_function = simple_loop_function
				else:
					loop_function = None

				# #################
				# # ATTENTION DECODER
				# #################
				# # First calculate a concatenation of encoder outputs to put attention on.
				# top_states = [array_ops.reshape(e, [-1, 1, wrapper_cell.output_size]) for e in encoder_outputs]
				# attention_states = array_ops.concat(1, top_states)


				# # return tf.nn.seq2seq.attention_decoder(decoder_inputs, enc_state, attention_states, wrapper_cell, output_size=self.output_size,loop_function=loop_function)

				# initial_state = enc_state
				# output_size = self.output_size
				# num_heads = 1
				# dtype = dtypes.float32
				# scope = None
				# initial_state_attention = False

				# if not decoder_inputs:
				# 	raise ValueError("Must provide at least 1 input to attention decoder.")
				# if num_heads < 1:
				# 	raise ValueError("With less than 1 heads, use a non-attention decoder.")
				# if not attention_states.get_shape()[1:2].is_fully_defined():
				# 	raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
				# 			% attention_states.get_shape())
				# if output_size is None:
				# 	output_size = wrapper_cell.output_size

				# with variable_scope.variable_scope(scope or "attention_decoder") as scope:
				# 	# dtype = scope.dtype

				# 	batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
				# 	attn_length = attention_states.get_shape()[1].value
				# 	attn_size = attention_states.get_shape()[2].value

				# 	# To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
				# 	hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
				# 	hidden_features = []
				# 	v = []
				# 	attention_vec_size = attn_size  # Size of query vectors for attention.
				# 	for a in xrange(num_heads):
				# 		k = variable_scope.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
				# 		hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
				# 		v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

				# 	state = initial_state

				# 	def attention(query):
				# 		"""Put attention masks on hidden using hidden_features and query."""
				# 		ds = []  # Results of attention reads will be stored here.
				# 		# if nest.is_sequence(query):  # If the query is a tuple, flatten it.
				# 		# 	query_list = nest.flatten(query)
				# 		# 	for q in query_list:  # Check that ndims == 2 if specified.
				# 		# 		ndims = q.get_shape().ndims
				# 		# 		if ndims:
				# 		# 			assert ndims == 2
				# 		# 	query = array_ops.concat(1, query_list)
				# 		for a in xrange(num_heads):
				# 			with variable_scope.variable_scope("Attention_%d" % a):
				# 				y = rnn_cell._linear(query, attention_vec_size, True)
				# 				y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
				# 				# Attention mask is a softmax of v^T * tanh(...).
				# 				s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
				# 				a = nn_ops.softmax(s)
				# 				# Now calculate the attention-weighted vector d.
				# 				d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
				# 				ds.append(array_ops.reshape(d, [-1, attn_size]))
				# 		return ds

				# 	outputs = []
				# 	prev = None
				# 	batch_attn_size = array_ops.pack([batch_size, attn_size])
				# 	attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
				# 				for _ in xrange(num_heads)]
				# 	for a in attns:  # Ensure the second shape of attention vectors is set.
				# 		a.set_shape([None, attn_size])
				# 	if initial_state_attention:
				# 		attns = attention(initial_state)
				# 	for i, inp in enumerate(decoder_inputs):
				# 		if i > 0:
				# 			variable_scope.get_variable_scope().reuse_variables()
				# 		# If loop_function is set, we use it instead of decoder_inputs.
				# 		if loop_function is not None and prev is not None:
				# 			with variable_scope.variable_scope("loop_function", reuse=True):
				# 				inp = loop_function(prev, i)
				# 		# Merge input and previous attentions into one vector of the right size.
				# 		input_size = inp.get_shape().with_rank(2)[1]
				# 		if input_size.value is None:
				# 			raise ValueError("Could not infer input size from input: %s" % inp.name)
				# 		x = rnn_cell._linear([inp] + attns, input_size, True)
				# 		# Run the RNN.
				# 		cell_output, state = wrapper_cell(x, state)
				# 		# Run the attention mechanism.
				# 		if i == 0 and initial_state_attention:
				# 			with variable_scope.variable_scope(variable_scope.get_variable_scope(),
				# 	                                       reuse=True):
				# 				attns = attention(state)
				# 		else:
				# 			attns = attention(state)

				# 		with variable_scope.variable_scope("AttnOutputProjection"):
				# 			output = rnn_cell._linear([cell_output] + attns, output_size, True)
				# 			output = tf.nn.sigmoid(output)
				# 		if loop_function is not None:
				# 			prev = output
				# 		outputs.append(output)

				# 	return outputs, state




				#################
				# BASIC DECODER
				#################
				with variable_scope.variable_scope("my_rnn_decoder"):
					state = enc_state
					outputs = []
					prev = None
					for i, inp in enumerate(decoder_inputs):
						if loop_function is not None and prev is not None:
							with variable_scope.variable_scope("loop_function", reuse=True):
								inp = loop_function(prev, i)
						if i > 0:
							variable_scope.get_variable_scope().reuse_variables()
						output, state = wrapper_cell(inp, state)
						output = tf.matmul(output, softmax_w) + softmax_b
						output = tf.nn.sigmoid(output)
						outputs.append(output)
						if loop_function is not None:
							prev = output
				
					return outputs, state





		# Feeds for inputs.
		self.encoder_inputs = []
		self.decoder_inputs = []
		for i in xrange(bucket[0]):
			self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, input_size], name="encoder{0}".format(i)))
		for i in xrange(bucket[1] + 1):
			self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[None, input_size], name="decoder{0}".format(i)))

		# Our targets are decoder inputs shifted by one.
		targets = [self.decoder_inputs[i + 1]
							 for i in xrange(len(self.decoder_inputs) - 1)]


		# Training outputs and losses.
		def softmax_loss_function(logit, target):
			# target = array_ops.reshape(target, [-1])

			# return tf.nn.softmax_cross_entropy_with_logits(logit, target)
			# return tf.nn.sigmoid_cross_entropy_with_logits(logit, target)

			# return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, target))

			return tf.squared_difference(target, logit)



		# my method
		with variable_scope.variable_scope("model_with_buckets"):
			bucket = self.bucket

			bucket_outputs, _ = seq2seq_f(self.encoder_inputs, self.decoder_inputs[:-1], forward_only)
			outputs = bucket_outputs

			bucket_outputs = bucket_outputs[:-1]
			val_targets = targets[:-1]
			# print("%%%%%%%%%%%%%%%%%%%%", len(bucket_outputs))
			# print("%%%%%%%%%%%%%%%%%%%%", len(val_targets))

			loss_list = []
			for logit, target in zip(bucket_outputs, val_targets):
				crossent = softmax_loss_function(logit, target)
				loss_list.append(crossent)
			loss = tf.add_n(loss_list)

			self.outputs = outputs
			self.loss = loss




		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)

			gradients = tf.gradients(self.loss, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

			self.gradient_norm = norm
			self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

			# _optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
			# self.update = _optimizer


		self.saver = tf.train.Saver(tf.all_variables())

	def step(self, session, encoder_inputs, decoder_inputs, forward_only):
		"""Run a step of the model feeding the given inputs.
		Args:
			session: tensorflow session to use.
			encoder_inputs: list of numpy int vectors to feed as encoder inputs.
			decoder_inputs: list of numpy int vectors to feed as decoder inputs.
			target_weights: list of numpy float vectors to feed as target weights.
			bucket_id: which bucket of the model to use.
			forward_only: whether to do the backward step or only forward.
		Returns:
			A triple consisting of gradient norm (or None if we did not do backward),
			average perplexity, and the outputs.
		Raises:
			ValueError: if length of encoder_inputs, decoder_inputs, or
				target_weights disagrees with bucket size for the specified bucket_id.
		"""
		# Check if the sizes match.
		encoder_size, decoder_size = self.bucket
		if len(encoder_inputs) != encoder_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
											 " %d != %d." % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError("Decoder length must be equal to the one in bucket,"
											 " %d != %d." % (len(decoder_inputs), decoder_size))

		# Input feed: encoder inputs, decoder inputs, as provided.
		input_feed = {}

		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]

		# Since our targets are decoder inputs shifted by one, we need one more.
		last_target = self.decoder_inputs[decoder_size].name
		# input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
		input_feed[last_target] = np.zeros([self.batch_size, self.input_size])

		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.update,  # Update Op that does SGD.
							self.gradient_norm,  # Gradient norm.
							self.loss]  # Loss for this batch.
			#调试 输出结果
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[l])
		else:
			output_feed = [self.loss]  # Loss for this batch.
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[l])

		outputs = session.run(output_feed, input_feed)
		# print (outputs)
		if not forward_only:
			return outputs[1], outputs[2], outputs[3:]  # Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

	# def get_batch(self, data, bucket_id):
	# 	"""Get a random batch of data from the specified bucket, prepare for step.
	# 	To feed data in step(..) it must be a list of batch-major vectors, while
	# 	data here contains single length-major cases. So the main logic of this
	# 	function is to re-index data cases to be in the proper format for feeding.
	# 	Args:
	# 		data: a tuple of size len(self.buckets) in which each element contains
	# 			lists of pairs of input and output data that we use to create a batch.
	# 		bucket_id: integer, which bucket to get the batch for.
	# 	Returns:
	# 		The triple (encoder_inputs, decoder_inputs, target_weights) for
	# 		the constructed batch that has the proper format to call step(...) later.
	# 	"""
	# 	encoder_size, decoder_size = self.buckets[bucket_id]
	# 	encoder_inputs, decoder_inputs = [], []

	# 	# Get a random batch of encoder and decoder inputs from data,
	# 	# pad them if needed, reverse encoder inputs and add GO to decoder.
	# 	for _ in xrange(self.batch_size):
	# 		# print(data[bucket_id][1])
	# 		# [encoder_input, decoder_input] = data[bucket_id][1]
	# 		encoder_input, decoder_input = random.choice(data[bucket_id])

	# 		encoder_inputs.append(list(reversed(encoder_input)))

	# 		decoder_inputs.append(decoder_input)

	# 	# Now we create batch-major vectors from the data selected above.
	# 	batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

	# 	# Batch encoder inputs are just re-indexed encoder_inputs.
	# 	for length_idx in xrange(encoder_size):
	# 		batch_encoder_inputs.append(
	# 				np.array([encoder_inputs[batch_idx][length_idx]
	# 									for batch_idx in xrange(self.batch_size)], dtype=np.float32))

	# 	# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
	# 	for length_idx in xrange(decoder_size):
	# 		batch_decoder_inputs.append(
	# 				np.array([decoder_inputs[batch_idx][length_idx]
	# 									for batch_idx in xrange(self.batch_size)], dtype=np.float32))

	# 		# Create target_weights to be 0 for targets that are padding.
	# 		batch_weight = np.ones([self.batch_size, self.input_size], dtype=np.float32)
	# 		# batch_weight = np.ones([self.batch_size], dtype=np.float32)
	# 		for batch_idx in xrange(self.batch_size):
	# 			# We set weight to 0 if the corresponding target is a PAD symbol.
	# 			# The corresponding target is decoder_input shifted by 1 forward.
	# 			if length_idx < decoder_size - 1:
	# 				target = decoder_inputs[batch_idx][length_idx + 1]
	# 			if length_idx == decoder_size - 1:
	# 				batch_weight[batch_idx] = np.zeros(self.input_size)
	# 		batch_weights.append(batch_weight)
	# 	# print(batch_encoder_inputs)
	# 	# print(batch_decoder_inputs)
	# 	# print(batch_weights)
	# 	return batch_encoder_inputs, batch_decoder_inputs, batch_weights


	# def get_batch_seq(self, seq_data, base_length):

		encoder_size, decoder_size = self.bucket
		encoder_inputs, decoder_inputs = [], []

		# Get a random batch of encoder and decoder inputs from data,
		# pad them if needed, reverse encoder inputs and add GO to decoder.
		unit_length = len(data[bucket_id][0])
		seq_length = (encoder_size+decoder_size)*base_length
		available_data_length = len(data[bucket_id]) - seq_length
		GO_ID = np.zeros([base_length*unit_length])
		# print("available_data_length",available_data_length)
		batch_counter = 0
		# print("self.batch_size",self.batch_size)
		while batch_counter < self.batch_size:
			# random choice a start
			seed = 1
			# seed = random.randint(0,available_data_length)
			sub_seq = data[bucket_id][seed:seed+seq_length]
			# if this seq is available and construct encoder_input and decoder_input
			seq_is_available = True
			bid_price_is_same = True
			ask_price_is_same = True
			compare_bid_price = float(sub_seq[0][0])
			compare_ask_price = float(sub_seq[0][int(unit_length/2)])
			group_counter = 0
			group_container = []
			input_seq = []
			for sec in sub_seq:
				bid_part = []
				for x in sec[:int(unit_length/2)]:
					x = float(x)
					if bid_price_is_same and x != compare_bid_price:
						bid_price_is_same = False
					v = float( '%0.5f'%( ((x-compare_bid_price)*scale+1.0)/2.0 ) )
					# ==================================
					# THIS CAN INSERT AN LIMIT FOR PRICES
					# ==================================
					if v > 1.0: 
						v = 1.0
					if v < 0.0: 
						v = 0.0
					# ==================================
					bid_part.append(v)
					# bid_part.append(0.67)

				ask_part = []
				for x in sec[int(unit_length/2):]:
					x = float(x)
					if ask_price_is_same and x != compare_ask_price:
						ask_price_is_same = False
					v = float( '%0.5f'%( ((x-compare_ask_price)*scale+1.0)/2.0 ) )
					# ==================================
					# THIS CAN INSERT AN LIMIT FOR PRICES
					# ==================================
					if v > 1.0:
						# print("A1") 
						v = 1.0
					if v < 0.0: 
						# print("B1") 
						v = 0.0
					# ==================================
					ask_part.append(v)
					# ask_part.append(0.67)

				group_container += bid_part+ask_part
				group_counter += 1
				# if group_counter > base_length:
				# 	group_container = group_container[unit_length*base_length:]
				if group_counter == base_length:
					input_seq.append(group_container)
					group_container = []
					group_counter = 0

				# break


			# add one batch
			if bid_price_is_same and ask_price_is_same:
				seq_is_available = False
				# print("******SAME PRICE*******")
			if seq_is_available:
				encoder_input = input_seq[:encoder_size]
				decoder_input = input_seq[encoder_size:-1]
				encoder_inputs.append(encoder_input)
				# encoder_inputs.append(list(reversed(encoder_input)))
				# decoder_inputs.append(decoder_input)
				decoder_inputs.append([GO_ID]+decoder_input)
				batch_counter += 1
				# print(len(encoder_input),len(decoder_input),batch_counter)

			# print("sub_seq[0]",sub_seq[0])
			# print("group_container",group_container)
			# print("len(input_seq)",len(input_seq))
			# print("len(encoder_input)",len(encoder_input))
			# print("len(decoder_input)",len(decoder_input))
			# print("len(decoder_input[0])",len(decoder_input[0]))

			# print(encoder_input)
			# print(decoder_input)

			# print("unit_length",unit_length)
			# print("seq_length",seq_length)
			# print("len(data[bucket_id])",len(data[bucket_id]))
			# print("available_data_length",available_data_length)
			# print("len(sub_seq)",len(sub_seq))

			# break


			

		# Now we create batch-major vectors from the data selected above.
		batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

		# Batch encoder inputs are just re-indexed encoder_inputs.
		for length_idx in xrange(encoder_size):
			batch_encoder_inputs.append(
					np.array([encoder_inputs[batch_idx][length_idx]
										for batch_idx in xrange(self.batch_size)], dtype=np.float32))

		# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
		for length_idx in xrange(decoder_size):
			batch_decoder_inputs.append(
					np.array([decoder_inputs[batch_idx][length_idx]
										for batch_idx in xrange(self.batch_size)], dtype=np.float32))

			# Create target_weights to be 0 for targets that are padding.
			batch_weight = np.ones([self.batch_size, self.input_size], dtype=np.float32)
			# batch_weight = np.ones([self.batch_size], dtype=np.float32)

			for batch_idx in xrange(self.batch_size):
				# We set weight to 0 if the corresponding target is a PAD symbol.
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < decoder_size - 1:
					target = decoder_inputs[batch_idx][length_idx + 1]
				if length_idx == decoder_size - 1:
					batch_weight[batch_idx] = np.zeros(self.input_size)

			batch_weights.append(batch_weight)
		# print(batch_encoder_inputs)
		# print(batch_decoder_inputs)
		# print(batch_weights)
		return batch_encoder_inputs, batch_decoder_inputs, batch_weights






def model_test():
	"""Test the s2s model."""
	with tf.Session() as sess:
		print("Model-test for s2s model.")
		bucket=(3, 3)

		# Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
		model = Seq2SeqModel(input_size=5, output_size=5, bucket=bucket, size=100, num_layers=2, max_gradient_norm=5.0, batch_size=5, learning_rate=0.5, learning_rate_decay_factor=0.9)

		sess.run(tf.initialize_all_variables())

		# Fake data set.
		data_set = []
		data_set_size = 50
		for _ in range(data_set_size):
			item = [random.random() for _ in range(model.input_size)]
			# item = [0.0 for _ in range(model.input_size)]
			# for _ in range(5):
			# 	item[random.randint(0, model.input_size-1)] = 1.0
			data_set.append(item)

		true_list = []
		error_list = []
		for _ in xrange(2001):  # Train the fake model.

			# data random choose
			encoder_inputs = [ [] for x in range(bucket[0]) ]
			decoder_inputs = [ [] for x in range(bucket[1]) ]
			for x in range(model.batch_size):
				seed = random.randint(0, data_set_size-1-bucket[0]-bucket[1])
				for i in range(bucket[0]):
					encoder_inputs[i].append(data_set[seed+i])
				for i in range(bucket[1]):
					decoder_inputs[i].append(data_set[seed+bucket[0]+i])

			decoder_inputs = decoder_inputs[:-1]
			decoder_inputs = [[[0.0 for x in range(model.input_size)] for x in range(model.batch_size)]] + decoder_inputs

			# print( np.array(encoder_inputs).shape )
			# print( np.array(decoder_inputs).shape )

			stepout = model.step(sess, encoder_inputs, decoder_inputs, False)

			
			p_out = np.array(stepout[2])
			t_out = np.array(decoder_inputs)

			# print(np.array(p_out).shape)
			# print(np.array(t_out).shape)

			p_out = p_out[:-1]
			t_out = t_out[1:]

			# print(p_out[0][0][:5])
			# print(t_out[0][0][:5])
			# error = np.average(np.absolute(t_out-p_out))
			



			# label_predict = ( np.argmax(p_out, axis=2) >= 0.5 ).astype(int)
			# label_target = ( np.argmax(t_out, axis=2) >= 0.5 ).astype(int)

			sub_error = np.average(np.absolute(np.array(p_out) - np.array(t_out)))
			error_list.append(sub_error)

			label_predict = ( np.array(p_out) >= 0.5 ).astype(int)
			label_target = ( np.array(t_out) >= 0.5 ).astype(int)

			# print(np.array(label_predict).shape)
			# print(np.array(label_target).shape)


			# print(label_predict)
			# print(label_target)

			results = np.equal(label_predict,label_target)
			# print results

			results = np.sum(results, axis=2)

			# print(results)

			results = (results == model.output_size).astype(int)

			# print results
			# print np.array(results).shape
			true_list.append(float(np.sum(results))/float((model.bucket[1]-1)*model.batch_size))
			

			if _ % 100 == 0:
				print("@@@@@@@@@@@@@@",_)
				print("LOSS",np.average(stepout[1]))
				print("True percent ", np.sum(true_list)/100.0)
				print("Error average", np.sum(error_list)/100.0)
				true_list = []
				error_list = []

			# break

if __name__ == "__main__":
	model_test()