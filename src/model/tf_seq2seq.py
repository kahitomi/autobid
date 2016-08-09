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

def attention_seq2seq(encoder_inputs, decoder_inputs, cell, feed_previous=False, dtype=dtypes.float32, scope=None):
	with variable_scope.variable_scope(scope or "attention_seq2seq"):

		# Encoder.
	    encoder_outputs, encoder_state = rnn.rnn(
	        cell, encoder_inputs, dtype=dtype)

	    # First calculate a concatenation of encoder outputs to put attention on.
	    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
	                  for e in encoder_outputs]
	    attention_states = array_ops.concat(1, top_states)

	    # if feed_previous
	    if feed_previous:
			def simple_loop_function(prev, _):
				# prev_symbol = math_ops.argmax(prev, 1)
				return prev
			loop_function = simple_loop_function
	    else:
	    	loop_function = None

	    return tf.nn.seq2seq.attention_decoder(decoder_inputs, encoder_state, attention_states, cell, loop_function=loop_function)





class Seq2SeqModel(object):
	"""Sequence-to-sequence model with attention and for multiple buckets.
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

	def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, use_lstm=False, num_samples=512, forward_only=False):

		"""Create the model.
		Args:
			source_vocab_size: size of the source vocabulary.
			target_vocab_size: size of the target vocabulary.
			buckets: a list of pairs (I, O), where I specifies maximum input length
				that will be processed in that bucket, and O specifies maximum output
				length. Training instances that have inputs longer than I or outputs
				longer than O will be pushed to the next bucket and padded accordingly.
				We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
			size: number of units in each layer of the model.
			num_layers: number of layers in the model.
			max_gradient_norm: gradients will be clipped to maximally this norm.
			batch_size: the size of the batches used during training;
				the model construction is independent of batch_size, so it can be
				changed after initialization if this is convenient, e.g., for decoding.
			learning_rate: learning rate to start with.
			learning_rate_decay_factor: decay learning rate by this much when needed.
			use_lstm: if true, we use LSTM cells instead of GRU cells.
			num_samples: number of samples for sampled softmax.
			forward_only: if set, we do not construct the backward pass in the model.
		"""
		self.source_vocab_size = source_vocab_size
		self.target_vocab_size = target_vocab_size
		self.buckets = buckets
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
				self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		

		# Create the internal multi-layer cell for our RNN.
		single_cell = tf.nn.rnn_cell.GRUCell(size)
		if use_lstm:
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
		cell = single_cell
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

		# The seq2seq function: we use the attention version.
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return attention_seq2seq(encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, feed_previous=do_decode)

		# Feeds for inputs.
		self.encoder_inputs = []
		self.decoder_inputs = []
		self.target_weights = []
		for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
			self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, source_vocab_size], name="encoder{0}".format(i)))
		for i in xrange(buckets[-1][1] + 1):
			self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[None, source_vocab_size], name="decoder{0}".format(i)))
			self.target_weights.append(tf.placeholder(tf.float32, shape=[None, source_vocab_size], name="weight{0}".format(i)))

		# Our targets are decoder inputs shifted by one.
		targets = [self.decoder_inputs[i + 1]
							 for i in xrange(len(self.decoder_inputs) - 1)]


		# Training outputs and losses.
		def softmax_loss_function(logit, target):
			# target = array_ops.reshape(target, [-1])
			return tf.nn.sigmoid_cross_entropy_with_logits(logit, target)
		if forward_only:
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
					self.encoder_inputs, self.decoder_inputs, targets,
					self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
					softmax_loss_function = softmax_loss_function)
		else:
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
					self.encoder_inputs, self.decoder_inputs, targets,
					self.target_weights, buckets,
					lambda x, y: seq2seq_f(x, y, False),
					softmax_loss_function = softmax_loss_function)

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			for b in xrange(len(buckets)):
				gradients = tf.gradients(self.losses[b], params)
				clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
				self.gradient_norms.append(norm)
				self.updates.append(opt.apply_gradients(
						zip(clipped_gradients, params), global_step=self.global_step))

		self.saver = tf.train.Saver(tf.all_variables())

	def step(self, session, encoder_inputs, decoder_inputs, target_weights,
					 bucket_id, forward_only):
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
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
											 " %d != %d." % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError("Decoder length must be equal to the one in bucket,"
											 " %d != %d." % (len(decoder_inputs), decoder_size))
		if len(target_weights) != decoder_size:
			raise ValueError("Weights length must be equal to the one in bucket,"
											 " %d != %d." % (len(target_weights), decoder_size))

		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		# print (type(encoder_inputs[0]))
		# print (len(encoder_inputs[0]))
		# print (encoder_inputs[0].shape)
		# print (encoder_inputs[0])
		# print (type(decoder_inputs[0]))
		# print (len(decoder_inputs[0]))
		# print (decoder_inputs[0].shape)
		# print (decoder_inputs[0])
		# print (type(target_weights[0]))
		# print (len(target_weights[0]))
		# print (target_weights[0].shape)
		# print (target_weights[0])
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
			input_feed[self.target_weights[l].name] = target_weights[l]

		# Since our targets are decoder inputs shifted by one, we need one more.
		last_target = self.decoder_inputs[decoder_size].name
		# input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
		input_feed[last_target] = np.zeros([self.batch_size, self.source_vocab_size])

		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
										 self.gradient_norms[bucket_id],  # Gradient norm.
										 self.losses[bucket_id]]  # Loss for this batch.

			# output_feed = [self.updates[bucket_id]]  # Loss for this batch.
		else:
			output_feed = [self.losses[bucket_id]]  # Loss for this batch.
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[bucket_id][l])

		outputs = session.run(output_feed, input_feed)
		# print (outputs)
		if not forward_only:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

	def get_batch(self, data, bucket_id):
		"""Get a random batch of data from the specified bucket, prepare for step.
		To feed data in step(..) it must be a list of batch-major vectors, while
		data here contains single length-major cases. So the main logic of this
		function is to re-index data cases to be in the proper format for feeding.
		Args:
			data: a tuple of size len(self.buckets) in which each element contains
				lists of pairs of input and output data that we use to create a batch.
			bucket_id: integer, which bucket to get the batch for.
		Returns:
			The triple (encoder_inputs, decoder_inputs, target_weights) for
			the constructed batch that has the proper format to call step(...) later.
		"""
		encoder_size, decoder_size = self.buckets[bucket_id]
		encoder_inputs, decoder_inputs = [], []

		# Get a random batch of encoder and decoder inputs from data,
		# pad them if needed, reverse encoder inputs and add GO to decoder.
		for _ in xrange(self.batch_size):
			encoder_input, decoder_input = random.choice(data[bucket_id])

			encoder_inputs.append(list(reversed(encoder_input)))

			decoder_inputs.append(decoder_input)

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
			batch_weight = np.ones([self.batch_size, self.source_vocab_size], dtype=np.float32)
			# batch_weight = np.ones([self.batch_size], dtype=np.float32)
			for batch_idx in xrange(self.batch_size):
				# We set weight to 0 if the corresponding target is a PAD symbol.
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < decoder_size - 1:
					target = decoder_inputs[batch_idx][length_idx + 1]
				# if length_idx == decoder_size - 1 or target == data_utils['PAD_ID']:
				# 	batch_weight[batch_idx] = np.zeros(self.source_vocab_size)
			batch_weights.append(batch_weight)
		return batch_encoder_inputs, batch_decoder_inputs, batch_weights


	def get_batch_seq(self, data, bucket_id, base_length, scale=2000.0):
		# self.source_vocab_size
		# self.target_vocab_size

		encoder_size, decoder_size = self.buckets[bucket_id]
		encoder_inputs, decoder_inputs = [], []

		# Get a random batch of encoder and decoder inputs from data,
		# pad them if needed, reverse encoder inputs and add GO to decoder.
		unit_length = len(data[bucket_id][0])
		seq_length = (encoder_size+decoder_size)*base_length
		available_data_length = len(data[bucket_id]) - seq_length

		batch_counter = 0
		# print("self.batch_size",self.batch_size)
		while batch_counter < self.batch_size:
			# random choice a start
			seed = random.randint(0,available_data_length)
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
					v = float( '%0.4f'%( ((x-compare_bid_price)*scale+1.0)/2.0 ) )
					# ==================================
					# THIS CAN INSERT AN LIMIT FOR PRICES
					# ==================================
					if v > 1.0: v = 1.0
					if v < 0.0: v = 0.0
					# ==================================
					bid_part.append(v)
				ask_part = []
				for x in sec[int(unit_length/2):]:
					x = float(x)
					if ask_price_is_same and x != compare_ask_price:
						ask_price_is_same = False
					v = float( '%0.4f'%( ((x-compare_ask_price)*scale+1.0)/2.0 ) )
					# ==================================
					# THIS CAN INSERT AN LIMIT FOR PRICES
					# ==================================
					if v > 1.0: v = 1.0
					if v < 0.0: v = 0.0
					# ==================================
					ask_part.append(v)
				group_container += bid_part+ask_part
				group_counter += 1
				if group_counter > base_length:
					group_container = group_container[unit_length*base_length:]
				if group_counter == base_length:
					input_seq.append(group_container)
					group_container = []
					group_counter = 0

				# break


			# add one batch
			if bid_price_is_same and ask_price_is_same:
				seq_is_available = False
			if seq_is_available:
				encoder_input = input_seq[:encoder_size]
				decoder_input = input_seq[encoder_size:]
				encoder_inputs.append(list(reversed(encoder_input)))
				decoder_inputs.append(decoder_input)
				batch_counter += 1


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
			batch_weight = np.ones([self.batch_size, self.source_vocab_size], dtype=np.float32)
			# batch_weight = np.ones([self.batch_size], dtype=np.float32)
			for batch_idx in xrange(self.batch_size):
				# We set weight to 0 if the corresponding target is a PAD symbol.
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < decoder_size - 1:
					target = decoder_inputs[batch_idx][length_idx + 1]
				# if length_idx == decoder_size - 1 or target == data_utils['PAD_ID']:
				# 	batch_weight[batch_idx] = np.zeros(self.source_vocab_size)
			batch_weights.append(batch_weight)
		return batch_encoder_inputs, batch_decoder_inputs, batch_weights