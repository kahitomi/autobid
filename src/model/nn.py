# -*- coding: utf-8 -*-
# Tensorflow 简单ann相关模型

import random

import numpy as np
import tensorflow as tf






class nn(object):
	"""The ANN model."""

	def __init__(self, is_training, batch_size, hidden_size, hidden_number, input_size, output_size, learning_rate=0.1):
		"""
			config
			
			steps 			seq长度

			batch_size		batch大小

			hidden_size 	隐层大小
			hidden_number	隐层数量

			input_size 		输入大小
			output_size		输出大小

			is_LSTM			是不是用LSTM, 默认是GRU

			learning_rate 	学习率
			is_training		是不是训练

		"""
		print "# 模型初始化中"

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.hidden_number = hidden_number
		self.input_size = input_size
		self.output_size = output_size

		# self.learning_rate = learning_rate
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)



		# 节点初始化
		cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
		if self.hidden_number > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.hidden_number)
		cell = tf.nn.rnn_cell.InputProjectionWrapper(cell, self.hidden_size, self.input_size)
		# cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.output_size)



		# 占位符初始化
		# self.targets = tf.placeholder(tf.float32, shape=[None, self.output_size])
		self.targets = (tf.placeholder(tf.float32, shape=[None, self.output_size]))
		self.inputs = (tf.placeholder(tf.float32, shape=[None, self.input_size]))
		

		cell_outputs, state = tf.nn.rnn(cell, [self.inputs], dtype=tf.float32)

		# output = tf.reshape(tf.concat(1, cell_outputs), [-1, self.hidden_size])
		softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.output_size])
		softmax_b = tf.get_variable("softmax_b", [self.output_size])
		# self.outputs = logits = tf.matmul(output, softmax_w) + softmax_b
		# self.outputs = logits = tf.matmul(cell_outputs[0], softmax_w) + softmax_b
		self.outputs = logits = tf.matmul(cell_outputs[0], softmax_w) + softmax_b
		# self.outputs = logits = cell_outputs[0]


		# loss 计算
		def softmax_loss_function(logit, target):

			# labels = tf.argmax(target, dimension=1)

			# return tf.nn.sparse_softmax_cross_entropy_with_logits(logit, labels)



			# return tf.reduce_mean(-tf.reduce_sum(logit * tf.log(target), reduction_indices=[1]))
			# return tf.nn.softmax_cross_entropy_with_logits(logit, target)


			# tf.greater_equal()
			return tf.nn.sigmoid_cross_entropy_with_logits(logit, target)

			# return tf.reduce_sum(tf.pow(logit-target, 2))/(2*self.batch_size)
			# return tf.squared_difference(target, logit)


		# # 现成函数losses
		# self.loss = tf.nn.seq2seq.sequence_loss(
		# 				logits,
		# 				self.targets,
		# 				# [tf.reshape(self.targets, [-1])],
		# 				tf.ones([self.steps, self.batch_size, self.output_size]),
		# 				softmax_loss_function = softmax_loss_function
		# 			)





		# 自定义losses
		losses = []

		crossent = softmax_loss_function(logits, self.targets)
		losses.append(crossent)

		self.loss = crossent
		# self.loss = tf.add_n(losses)
		# self.loss = losses







		# 网络学习
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


		# tvars = tf.trainable_variables()
		# grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
		# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).apply_gradients(zip(grads, tvars))




		return None



	# def load_all_data(self, inputs_outputs):
	# 	"""
	# 		inputs_outputs 	: list pairs of inputs and outputs
	# 		load all inputs_outputs
	# 	"""
	# 	random_seed_limit = len(inputs_outputs)
	# 	data = []
	# 	for x in range(len(inputs_outputs)):
	# 		self.inputs[x].



	def step(self, sess, inputs, outputs, is_training=True):
		# 训练一步
		feed_dict = {}

		# 添加输入输出
		feed_dict[self.inputs.name] = np.array(inputs)
		feed_dict[self.targets.name] = np.array(outputs)

		if is_training:
			predict_out = sess.run([self.loss, self.outputs, self.optimizer], feed_dict)
			# predict_out = sess.run([self.loss, self.optimizer], feed_dict)
		else:
			predict_out = sess.run(self.outputs, feed_dict)

		return predict_out




# 模型测试
def nn_test():
	print "ANN模型测试"

	with tf.Session() as sess:
		model = nn( batch_size=10, hidden_size=1000, hidden_number=2, input_size=100, output_size=20, is_training=True)
		sess.run(tf.initialize_all_variables())

		# 生成随机数据
		data_set = []
		for x in range(500): # 数据数量
			# 随机数
			# item = [random.random() for _ in range(model.input_size)]

			# 随机标签
			item_in = [0.0 for _ in range(model.input_size)]
			for _ in range(3):
				item_in[random.randint(0, model.input_size-1)] = 1.0

			item_out = [0.0 for _ in range(model.output_size)]
			for _ in range(3):
				item_out[random.randint(0, model.output_size-1)] = 1.0

			data_set.append([item_in, item_out])

		# 训练循环
		for _ in range(2001): # 循环数量
			# 随机抽取数据
			inputs = []
			outputs = []
			seed_limit = len(data_set)
			for x in range(model.batch_size):
				seed = random.randint(0, seed_limit-1)
				inputs.append(data_set[seed][0])
				outputs.append(data_set[seed][1])

			step_out = model.step(sess, inputs, outputs)

			if _ % 100 == 0:
				# print step_out
				print "============================"
				print "Epoch", _, ":",np.average(np.array(step_out[0]))
				# print np.array(step_out[0])
				# break
				step_out = model.step(sess, inputs, outputs, False)

				# print np.array(step_out).shape

				label_predict = (step_out >= 0.5).astype(int)
				label_target = outputs
				# print np.array(label_predict).shape
				# print np.array(label_target).shape
				# print label_predict[0]
				# print label_target[:0]

				results = np.equal(label_predict,label_target)
				# print results
				results = np.sum(results, axis=1)
				# print results
				results = (results == model.output_size).astype(int)
				# print results
				# print np.array(results).shape
				print "True Number",np.sum(results),"/", model.batch_size
				# numpy.count_nonzero(boolarr)

				# break


if __name__ == "__main__":
	nn_test()