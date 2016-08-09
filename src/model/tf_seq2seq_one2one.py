# -*- coding: utf-8 -*-
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random,datetime

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope

from IAS import word2vec

# 自己定义的网络模型
from IAS import _tf_seq2seq as tf_seq2seq

# # from tensorflow.models.rnn.translate import data_utils
# LDIC = len(word2vec.my_dictionary.token2id)

# data_utils = {
# 	# 'PAD_ID': np.zeros(300).tolist(),
# 	# 'GO_ID': np.zeros(300).tolist()
# 	'PAD_ID': LDIC+11,
# 	'GO_ID': LDIC+12,
# 	'EOS': LDIC + 10,
# 	'EOT': LDIC + 9,
# 	'UNK': LDIC + 999
# }

TASK_NUM = 1


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

		# with tf.device("/job:saver"):
		with tf.device("/job:worker"):
			self.source_vocab_size = source_vocab_size
			self.target_vocab_size = target_vocab_size
			self.buckets = buckets
			self.batch_size = batch_size
			self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
			self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
			self.global_step = tf.Variable(0, trainable=False)

			# word embedding function
			with tf.device("/cpu:0"):
				self.embedding = variable_scope.get_variable("embedding",[self.target_vocab_size, size])

			# If we usampled softmaxse , we need an output projection.
			output_projection = None
			softmax_loss_function = None
			# Sampled softmax only makes sense if we sample less than vocabulary size.
			with tf.device("/cpu:0"):
				w = tf.get_variable("proj_w", [size, self.target_vocab_size])
				w_t = tf.transpose(w)
				b = tf.get_variable("proj_b", [self.target_vocab_size])
				output_projection = (w, b)

			# Create the internal multi-layer cell for our RNN.
			single_cell = tf.nn.rnn_cell.GRUCell(size)
			if use_lstm:
				single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
			cell = single_cell
			if num_layers > 1:
				cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)



			# The seq2seq function: we use embedding for the input and attention.
			def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
				return tf.nn.seq2seq.embedding_attention_seq2seq(
						encoder_inputs, decoder_inputs, cell,
						num_encoder_symbols=source_vocab_size,
						num_decoder_symbols=target_vocab_size,
						embedding_size=size,
						output_projection=output_projection,
						feed_previous=do_decode)

			self.seq2seq_f = seq2seq_f



			self.encoder_inputs_list = []
			self.decoder_inputs_list = []
			self.target_weights_list = []
			self.targets_list = []

			self.outputs_list = []
			self.losses_list = []
			self.updates_list = []
			self.gradient_norms_list = []







			self.encoder_inputs = []
			self.decoder_inputs = []
			self.target_weights = []
			for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
				self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
			for i in xrange(buckets[-1][1] + 1):
				self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
				self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

			# Our targets are decoder inputs shifted by one.
			targets = [self.decoder_inputs[i + 1]
					for i in xrange(len(self.decoder_inputs) - 1)]

			def sampled_loss(inputs, labels):
				with tf.device("/cpu:0"):
					# labels = tf.reshape(labels, [-1])
					labels = tf.reshape(labels, [-1, 1])
					return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.target_vocab_size)
			softmax_loss_function = sampled_loss

		with tf.device("/job:worker"):

			# Training outputs and losses.
			if forward_only:
				self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
						self.encoder_inputs, self.decoder_inputs, targets,
						self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
						softmax_loss_function=softmax_loss_function
						)
				# If we use output projection, we need to project outputs for decoding.
				if output_projection is not None:
					for b in xrange(len(buckets)):
						self.outputs[b] = [
								tf.matmul(output, output_projection[0]) + output_projection[1]
								for output in self.outputs[b]
						]
			else:
				self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
						self.encoder_inputs, self.decoder_inputs, targets,
						self.target_weights, buckets,
						lambda x, y: seq2seq_f(x, y, False),
						softmax_loss_function=softmax_loss_function
						)

			# Gradients and SGD update operation for training the model.
			params = tf.trainable_variables()
			if not forward_only:
				self.gradient_norms = []
				self.updates = []
				opt = tf.train.GradientDescentOptimizer(self.learning_rate)
				for b in xrange(len(buckets)):
					# gradients = tf.gradients(self.losses[b], params, aggregation_method=2)
					gradients = tf.gradients(self.losses[b], params)
					clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
					self.gradient_norms.append(norm)
					self.updates.append(opt.apply_gradients(
						zip(clipped_gradients, params), global_step=self.global_step))


		# # 多机版本
		# for task_num in range(TASK_NUM):
		# 	# Feeds for inputs.
		# 	encoder_inputs = []
		# 	decoder_inputs = []
		# 	target_weights = []
		# 	for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
		# 		encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, size], name="encoder{0}.task{1}".format(i, task_num)))
		# 	for i in xrange(buckets[-1][1] + 1):
		# 		decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}.task{1}".format(i, task_num)))
		# 		target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}.task{1}".format(i, task_num)))

		# 	# Our targets are decoder inputs shifted by one.
		# 	targets = [decoder_inputs[i + 1]
		# 			for i in xrange(len(decoder_inputs) - 1)]

		# 	self.encoder_inputs_list.append(encoder_inputs)
		# 	self.decoder_inputs_list.append(decoder_inputs)
		# 	self.target_weights_list.append(target_weights)
		# 	self.targets_list.append(targets)


		# 	with tf.device("/job:worker/task:"+str(task_num)):
		# 		# with tf.variable_scope('task_%d' % task_num):
		# 		# with tf.variable_scope('task_all'):

		# 		if task_num>0:
		# 			tf.get_variable_scope().reuse_variables()

		# 		def sampled_loss(inputs, labels):
		# 			# with tf.device("/cpu:0"):
		# 			labels = tf.reshape(labels, [-1])
		# 			# labels = tf.reshape(labels, [-1, 1])
		# 			return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.target_vocab_size)
		# 		softmax_loss_function = sampled_loss

		# 		# Training outputs and losses.
		# 		if forward_only:
		# 			outputs, losses = tf.nn.seq2seq.model_with_buckets(
		# 					encoder_inputs, decoder_inputs, targets,
		# 					target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
		# 					# softmax_loss_function=softmax_loss_function
		# 					)
		# 			# If we use output projection, we need to project outputs for decoding.
		# 			if output_projection is not None:
		# 				for b in xrange(len(buckets)):
		# 					outputs[b] = [
		# 							tf.matmul(output, output_projection[0]) + output_projection[1]
		# 							for output in outputs[b]
		# 					]
		# 		else:
		# 			outputs, losses = tf.nn.seq2seq.model_with_buckets(
		# 					encoder_inputs, decoder_inputs, targets,
		# 					target_weights, buckets,
		# 					lambda x, y: seq2seq_f(x, y, False),
		# 					# softmax_loss_function=softmax_loss_function
		# 					)
		# 		self.outputs_list.append(outputs)
		# 		self.losses_list.append(losses)


		# 		# Gradients and SGD update operation for training the model.
		# 		params = tf.trainable_variables()
		# 		if not forward_only:
		# 			gradient_norms = []
		# 			updates = []
		# 			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
		# 			for b in xrange(len(buckets)):
		# 				gradients = tf.gradients(losses[b], params, aggregation_method=2)
		# 				clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
		# 				gradient_norms.append(norm)
		# 				updates.append(opt.apply_gradients(
		# 					zip(clipped_gradients, params), global_step=self.global_step))
		# 			self.gradient_norms_list.append(gradient_norms)
		# 			self.updates_list.append(updates)


		# Update all
		# self.update_bid = tf.placeholder(tf.int32, name="update_bid")
		# def update_f(bid):
		# 	counter = 0
		# 	for t in self.losses_list:
		# 		if counter == 0:
		# 			loss_changed_sum = t[bid]
		# 		else:
		# 			loss_changed_sum = tf.add(t[bid], loss_changed_sum)
		# 		counter += 1
		# 	return loss_changed_sum
		# self.total_loss = update_f


		# loss_changed_list = [t[self.update_bid] for t in self.losses_list]

		# gradients = tf.gradients(losses[self.update_bid], params)
		# clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
		# gradient_norms.append(norm)
		# updates.append(opt.apply_gradients(
		# 	zip(clipped_gradients, params), global_step=self.global_step))

		# self.updates_all = 

		with tf.device("/job:saver"):
			self.bug_fixer = tf.Variable(tf.zeros([10]), name="bug_fixer")
			self.saver = tf.train.Saver(tf.all_variables(), sharded=True)

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
		# if len(encoder_inputs[0]) != encoder_size:
		# 	raise ValueError("Encoder length must be equal to the one in bucket,"
		# 									 " %d != %d." % (len(encoder_inputs), encoder_size))
		# if len(decoder_inputs[0]) != decoder_size:
		# 	raise ValueError("Decoder length must be equal to the one in bucket,"
		# 									 " %d != %d." % (len(decoder_inputs), decoder_size))
		# if len(target_weights[0]) != decoder_size:
		# 	raise ValueError("Weights length must be equal to the one in bucket,"
		# 									 " %d != %d." % (len(target_weights), decoder_size))

		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
			input_feed[self.target_weights[l].name] = target_weights[l]

		# Since our targets are decoder inputs shifted by one, we need one more.
		last_target = self.decoder_inputs[decoder_size].name
		input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

		# for task_num in range(TASK_NUM):
		# 	# print (decoder_inputs[task_num])
		# 	for l in xrange(encoder_size):
		# 		input_feed[self.encoder_inputs_list[task_num][l].name] = encoder_inputs[task_num][l]
		# 	for l in xrange(decoder_size):
		# 		input_feed[self.decoder_inputs_list[task_num][l].name] = decoder_inputs[task_num][l]
		# 		input_feed[self.target_weights_list[task_num][l].name] = target_weights[task_num][l]

		# 	# Since our targets are decoder inputs shifted by one, we need one more.
		# 	last_target = self.decoder_inputs_list[task_num][decoder_size].name
		# 	input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
										 self.gradient_norms[bucket_id],  # Gradient norm.
										 self.losses[bucket_id]]  # Loss for this batch.


			# output_feed = []
			# for x in range(TASK_NUM):
			# 	output_feed.append(self.updates_list[x][bucket_id])
			# 	output_feed.append(self.gradient_norms_list[x][bucket_id])
			# 	output_feed.append(self.losses_list[x][bucket_id])

			# 	# outputs = session.run(output_feed, input_feed)

		else:
			output_feed = [self.losses[bucket_id]]  # Loss for this batch.
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[bucket_id][l])

		# output_feed.append(self.total_loss(bucket_id))
		# print(input_feed)
		# print(output_feed)
		outputs = session.run(output_feed, input_feed)
		# print ("output length ",len(outputs))
		# print (outputs)
		if not forward_only:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			# return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs.
			return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


	def get_batch(self, data, bucket_id, session):
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

			# print(encoder_input)

			# Encoder inputs are padded and then reversed.

			pad_id = word2vec.data_utils["PAD_ID"]

			encoder_pad = [pad_id] * (encoder_size - len(encoder_input))
			sub_encoder_inputs = list(reversed(encoder_input.tolist() + encoder_pad))
			# encoder_inputs.append(list(reversed(encoder_input.tolist() + encoder_pad)))
			encoder_inputs.append(sub_encoder_inputs)

			# Decoder inputs get an extra "GO" symbol, and are padded then.
			decoder_pad_size = decoder_size - len(decoder_input) - 1
			decoder_inputs.append([word2vec.data_utils["GO_ID"]] + decoder_input.tolist() + [pad_id] * decoder_pad_size)

		# Now we create batch-major vectors from the data selected above.
		batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

		# print (encoder_inputs[0])
		# Batch encoder inputs are just re-indexed encoder_inputs.
		for length_idx in xrange(encoder_size):
			batch_encoder_inputs.append(
					np.array(
						[encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)] 
					,dtype=np.int32)
					)

		# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
		for length_idx in xrange(decoder_size):
			batch_decoder_inputs.append(
					np.array([decoder_inputs[batch_idx][length_idx]
										for batch_idx in xrange(self.batch_size)], dtype=np.int32))

			# Create target_weights to be 0 for targets that are padding.
			batch_weight = np.ones(self.batch_size, dtype=np.float32)
			for batch_idx in xrange(self.batch_size):
				# We set weight to 0 if the corresponding target is a PAD symbol.
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < decoder_size - 1:
					target = decoder_inputs[batch_idx][length_idx + 1]
				# print (length_idx == decoder_size - 1, target == word2vec.data_utils["PAD_ID"])
				if length_idx == decoder_size - 1 or target == word2vec.data_utils["PAD_ID"]:
					batch_weight[batch_idx] = 0.0
			batch_weights.append(batch_weight)
		# print(14,datetime.datetime.today())
		# print ("batch_decoder_inputs",len(batch_decoder_inputs[0]))
		# print(batch_weights)
		# print("$$$$$$$$$$$$$$$$$$$$$")
		# print(batch_encoder_inputs)
		# print(batch_decoder_inputs)
		# print(batch_weights)
		return batch_encoder_inputs, batch_decoder_inputs, batch_weights