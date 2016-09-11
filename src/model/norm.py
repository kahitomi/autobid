# -*- coding: utf-8 -*-
# norm 用方法

import numpy as np


def tanh(array, range_number, compress=2.0):
	output = array/float(range_number)
	output = (np.tanh(compress*output)+1.0)/2.0
	return output