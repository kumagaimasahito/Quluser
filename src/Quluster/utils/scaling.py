import numpy as np

def min_max(x, axis=None):
	x_min = x.min(axis=axis, keepdims=True)
	x_max = x.max(axis=axis, keepdims=True)
	return (x - x_min) / (x_max - x_min)

def standardization(x, axis=None, ddof=0):
	x_mean = x.mean(axis=axis, keepdims=True)
	x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
	return (x - x_mean) / x_std