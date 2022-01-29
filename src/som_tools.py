import matplotlib.pyplot as plt

import numpy as np
import random as rd
import math


class SOMUnit:

	def __init__(self, x, y, input_lenght, initial_learning_rate = .1, learning_rate_decay = 10, initial_radius = 1, radius_decay = 10):
		self.weights = np.random.rand(input_lenght)
		self.x = x
		self.y = y

		self.initial_radius = initial_radius
		self.radius_decay = (radius_decay if radius_decay != 0 else 1)
		self.learning_rate_decay = (learning_rate_decay if learning_rate_decay != 0 else 1)
		self.initial_learning_rate = initial_learning_rate

	def calculate_distance(self, input_vector):
		return math.sqrt(np.sum((self.weights-input_vector)**2))

	def adjust_weights(self, input_vector, epoch, distance_matrix, best_unit):
		self.weights += self._learning_rate(epoch) * self._topological_distance(epoch, distance_matrix, best_unit) * (input_vector - self.weights)

	def _topological_distance(self, epoch, distance_matrix, best_unit):
		return math.exp(-(distance_matrix.get_distance_by_coords(self.x, self.y, best_unit.x, best_unit.y)**2)/(2 * self._neighborhood_size(epoch) **2))

	def _neighborhood_size(self, epoch):
		return self.initial_radius * math.exp(-(epoch)/(self.radius_decay))

	def _learning_rate(self, epoch):
		return self.initial_learning_rate * math.exp(-(epoch)/(self.learning_rate_decay))

	def get_weights(self):
		return self.weights

class MnemonicSOM:

	def __init__(self, active_unit_matrix, distance_matrix, input_lenght, initial_learning_rate = .1, learning_rate_decay = 10, initial_radius = 1, radius_decay = 10):
		self.active_unit_matrix = active_unit_matrix
		self.distance_matrix = distance_matrix

		self.initial_radius = initial_radius
		self.radius_decay = (radius_decay if radius_decay != 0 else 1)
		self.learning_rate_decay = (learning_rate_decay if learning_rate_decay != 0 else 1)
		self.initial_learning_rate = initial_learning_rate

		self.input_lenght = input_lenght

		self.units = []

		for x, line in enumerate(active_unit_matrix):
			unit_line = []
			for y, state in enumerate(line):
				if state == 1:
					unit_line.append(SOMUnit(x, y, input_lenght, 
						initial_learning_rate = self.initial_learning_rate, 
						learning_rate_decay = self.learning_rate_decay, 
						initial_radius = self.initial_radius, 
						radius_decay = self.radius_decay))
				else:
					unit_line.append(None)
			self.units.append(unit_line)

	def train(self, X, epochs):
		for epoch in range(1, epochs+1):

			print("training epoch: ", epoch, "                          ")
			input_set = X.copy()

			while len(input_set) > 0:

				# take random input
				i = rd.randint(0, len(input_set)-1)
				vec = input_set.pop(i)

				# find best unit
				best_unit = self._get_best_unit(vec)

				# adjust weights
				for line in self.units:
					for unit in line:
						if unit:
							unit.adjust_weights(vec, epoch, self.distance_matrix, best_unit)

				print("vectors left: ", len(input_set), "                          ", end = "\r")

		print("")


	def _get_best_unit(self, input_vector):
		best_unit = None
		best_value = -1

		for line in self.units:
			for unit in line:
				if unit:
					value = unit.calculate_distance(input_vector)
					if best_value == -1 or value < best_value:
						best_unit = unit
						best_value = value
					
		return best_unit

	# testing WIP
	def show_som(self, X, y):
		som_image = np.full((len(self.units), len(self.units[0])), 0)

		for vec, label in zip(X, y):
			unit = self._get_best_unit(vec)

			som_image[unit.y, unit.x] = label

		som_image = som_image/np.max(som_image)

		plt.imshow(np.clip(som_image, a_min = 0, a_max = 1))
		plt.show()

	# testing WIP
	def show_som2(self, X):
		som_image = np.full((len(self.units), len(self.units[0]), 3), 0.0)

		for x in range(som_image.shape[1]):
			for y in range(som_image.shape[0]):
				if self.units[x][y]:
					som_image[y, x] = self.units[x][y].weights

		plt.imshow(np.clip(som_image, a_min = 0, a_max = 1))
		plt.show()

	# testing WIP
	def show_som_multiple(self, X, y, ax):
		som_image = np.full((len(self.units), len(self.units[0])), 0)

		for vec, label in zip(X, y):
			unit = self._get_best_unit(vec)

			som_image[unit.y, unit.x] = label

		som_image = som_image/np.max(som_image)

		ax.imshow(np.clip(som_image, a_min = 0, a_max = 1))

	def export_weightlist(self):
		ret = {}

		ret["xdim"] = len(self.units)
		ret["ydim"] = len(self.units[0])
		ret["vec_dim"] = self.input_lenght

		ret["arr"] = []
		for x in range(len(self.units)):
			for y in range(len(self.units[0])):
				print(x, y)
				if self.units[x][y]:
					ret["arr"].append(self.units[x][y].get_weights())
				else:
					ret["arr"].append(np.full(self.input_lenght, -np.inf))

		return ret

class DistanceMatrix:

	def __init__(self, active_unit_matrix):
		self.unit_index = []

		for x in range(active_unit_matrix.shape[0]):
			for y in range(active_unit_matrix.shape[1]):
				if active_unit_matrix[x, y] == 1:
					self.unit_index.append([x, y])

		self.distance_array = np.full((len(self.unit_index), len(self.unit_index)), np.inf)

	def _get_index(self, x, y):
		for i, elem in enumerate(self.unit_index):
			if x == elem[0] and y == elem[1]:
				return i
		return -1

	def set_distance_by_index(self, i0, i1, value):
		self.distance_array[i0, i1] = value
		self.distance_array[i1, i0] = value

	def get_distance_by_index(self, i0, i1):
		return self.distance_array[i0, i1]

	def set_distance_by_coords(self, x0, y0, x1, y1, value):
		i0 = self._get_index(x0, y0)
		i1 = self._get_index(x1, y1)

		if i0 == -1 or i1 == -1:
			return

		return self.set_distance_by_index(i0, i1, value)

	def get_distance_by_coords(self, x0, y0, x1, y1):
		i0 = self._get_index(x0, y0)
		i1 = self._get_index(x1, y1)

		if i0 == -1 or i1 == -1:
			return np.inf

		return self.get_distance_by_index(i0, i1)

	def get_active_unit_count(self):
		return len(self.unit_index)

	def get_active_units(self):
		return self.unit_index

	def show_distance_map(self, image_width, image_height, x, y, normalize = False):
		distance_map = np.full((image_height, image_width), 0)

		for xp in range(distance_map.shape[1]):
			for yp in range(distance_map.shape[0]):
				dist = self.get_distance_by_coords(x, y, xp, yp)

				distance_map[yp, xp] = (dist if dist != np.inf else 0)

		if normalize:
			distance_map = distance_map/np.max(distance_map)

		plt.imshow(distance_map)
		plt.show()