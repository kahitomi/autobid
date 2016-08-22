# -*- coding: utf-8 -*-
# Tensorflow rnn相关模型

import random

import numpy as np
import tensorflow as tf



def sequence_loss(logits, targets):
	pass






class rnn(object):
	"""The NN model."""

	def __init__(self, is_training, steps, batch_size, hidden_size, hidden_number, input_size, output_size, is_LSTM=False, learning_rate=0.1):
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

		self.steps = steps
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.hidden_number = hidden_number
		self.input_size = input_size
		self.output_size = output_size
		self.is_LSTM = is_LSTM

		# self.learning_rate = learning_rate
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)



		# 节点初始化
		if is_LSTM:
			cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
		else:
			cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
		if self.hidden_number > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.hidden_number)
		cell = tf.nn.rnn_cell.InputProjectionWrapper(cell, self.hidden_size, self.input_size)
		# cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.output_size)



		# 占位符初始化
		# self.targets = tf.placeholder(tf.float32, shape=[None, self.output_size])
		self.targets = []
		self.inputs = []
		for x in range(self.steps):
			self.targets.append(tf.placeholder(tf.float32, shape=[None, self.output_size]))
			self.inputs.append(tf.placeholder(tf.float32, shape=[None, self.input_size]))

		outputs, state = tf.nn.rnn(cell, self.inputs, dtype=tf.float32)

		self.outputs = logits = outputs


		softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.output_size])
		softmax_b = tf.get_variable("softmax_b", [self.output_size])

		logits = []
		for output in outputs:
			logit = tf.matmul(output, softmax_w) + softmax_b
			logits.append(logit)

		self.logits = logits


		# output = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_size])
		# # self._output = outputs
		# # self.outputs = logits = tf.matmul(output, softmax_w) + softmax_b
		# self._logits = tf.matmul(output, softmax_w) + softmax_b


		# loss 计算
		def softmax_loss_function(logit, target):
			# return tf.reduce_mean(-tf.reduce_sum(logit * tf.log(target), reduction_indices=[1]))
			# return tf.nn.softmax_cross_entropy_with_logits(logit, target)
			return tf.nn.sigmoid_cross_entropy_with_logits(logit, target)

			# return tf.reduce_sum(tf.pow(logit-target, 2))/(2*self.batch_size)
			# return tf.squared_difference(target, logit)


		# # 现成函数losses
		# self.loss = tf.nn.seq2seq.sequence_loss(
		# 				logits,
		# 				self.targets,
		# 				# [tf.reshape(self.targets, [-1])],
		# 				np.ones([self.steps, self.batch_size]),
		# 				softmax_loss_function = softmax_loss_function
		# 			)





		# 自定义losses
		losses = []

		# _targets = (tf.reshape(tf.concat(1, self.targets), [-1, self.hidden_size]))
		# crossent = softmax_loss_function(self._logits, _targets)
		# losses.append(crossent)


		for logit, target in zip(logits, self.targets):
			crossent = softmax_loss_function(logit, target)
			losses.append(crossent)



		self.loss = tf.add_n(losses)
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

		# 添加输入
		input_steps = [[] for _ in range(self.steps)]
		for x in range(self.batch_size):
			for y in range(self.steps):
				input_steps[y].append(inputs[x][y])
		for x in range(self.steps):
			# print "~~",np.array(input_steps[x]).shape
			feed_dict[self.inputs[x].name] = np.array(input_steps[x])

		# 添加输出目标
		output_steps = [[] for _ in range(self.steps)]
		for x in range(self.batch_size):
			for y in range(self.steps):
				output_steps[y].append(outputs[x][y])
		for x in range(self.steps):
			# print "~~",np.array(output_steps[x]).shape
			feed_dict[self.targets[x].name] = np.array(output_steps[x])
		# feed_dict[self.targets.name] = np.array(outputs)

		if is_training:
			# predict_out = sess.run([self.loss, self.outputs, self.optimizer], feed_dict)
			predict_out = sess.run([self.loss, self.optimizer], feed_dict)
			# print np.average(np.array(predict_out[0]).shape)
		else:
			predict_out = sess.run(self.logits, feed_dict)
			# predict_out = sess.run(self.outputs, feed_dict)

		return predict_out




# 模型测试
def rnn_test():
	print "RNN模型测试"

	with tf.Session() as sess:
		model = rnn(steps=3, batch_size=2, hidden_size=300, hidden_number=2, input_size=30, output_size=30, is_training=True)
		sess.run(tf.initialize_all_variables())

		# 生成随机数据
		data_set = []
		for x in range(50): # 数据数量
			item = []
			for i in range(model.steps+1):
				# 随机数
				# item.append([random.random() for _ in range(model.input_size)])

				# 随机标签
				v_list = [0.0 for _ in range(model.input_size)]
				for _ in range(5):
					v_list[random.randint(0, model.input_size-1)] = 1.0
				item.append(v_list)


			data_set.append(item)

		# 训练循环
		for _ in range(2001): # 循环数量
			# 随机抽取数据
			inputs = []
			outputs = []
			seed_limit = len(data_set)
			for x in range(model.batch_size):
				seed = random.randint(0, seed_limit-1)
				inputs.append(data_set[seed][:model.steps])
				outputs.append(data_set[seed][1:])


			step_out = model.step(sess, inputs, outputs)

			if _ % 100 == 0:
				# print step_out
				print "============================"
				print "Epoch", _, ":",np.average(np.array(step_out[0]))
				# print np.array(step_out[0])
				# break
				step_out = model.step(sess, inputs, outputs, False)

				# print np.array(step_out).shape

				output_steps = [[] for _ in range(model.steps)]
				for x in range(model.batch_size):
					for y in range(model.steps):
						output_steps[y].append(outputs[x][y])

				# print np.average( np.absolute(np.array(step_out) - np.array(output_steps)))
				# print np.array(step_out).shape
				# print np.array(outputs).shape
				# print np.array(output_steps).shape
				label_predict = ( np.array(step_out) >= 0.5 ).astype(int)
				label_target = np.array(output_steps)
				print np.array(label_predict).shape
				print np.array(label_target).shape
				# print label_predict[:5]
				# print label_target[:5]

				# results = np.equal(label_predict,label_target)
				# print "True Number",np.sum(results),"/", model.steps*model.batch_size
				# # numpy.count_nonzero(boolarr)


				results = np.equal(label_predict,label_target)
				# print results
				results = np.sum(results, axis=2)
				print results
				results = (results == model.output_size).astype(int)
				# print results
				# print np.array(results).shape
				print "True Number",np.sum(results),"/", model.steps*model.batch_size

				# break


if __name__ == "__main__":
	rnn_test()