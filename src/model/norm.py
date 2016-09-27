# -*- coding: utf-8 -*-
# norm 用方法

import numpy as np


def tanh(array, range_number, compress=2.0):
	output = array/float(range_number)
	output = (np.tanh(compress*output)+1.0)/2.0
	return output


def tanh_decode(array, range_number, compress=2.0):
	output = array*2.0-1.0
	output[output > 0.999] = 0.999
	output[output < -0.999] = -0.999
	output = np.arctanh( output ) / compress
	output = output*range_number

	return output