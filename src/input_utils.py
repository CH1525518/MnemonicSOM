import numpy as np
import cv2
import som_tools

def read_dataset(data_path, name):
	X = []
	y = []

	with open(data_path + name + "/" + name + ".vec", "r") as vector_file:
		for line in vector_file:
			if line.startswith("$"):
				continue

			X.append(np.array(line.strip().split(" ")[:-1]).astype(np.float))

	with open(data_path + name + "/" + name + ".cls", "r") as class_file:
		for line in class_file:
			if line.startswith("$"):
				continue

			y.append(line.strip().split(" ")[1])

	return X, y

def convert_to_binary(image):
	return np.amax(image, axis=2)//255

def is_active_unit(image, x, y, width, height):
	x0 = min(x, image.shape[1])
	y0 = min(y, image.shape[0])
	x1 = min(x + width, image.shape[1])
	y1 = min(y + height, image.shape[0])

	return (np.sum(image[y0:y1, x0:x1])/image[y0:y1, x0:x1].size) > 0.5

def convert_to_active_unit_matrix(image, som_width, som_height):
	unit_width = image.shape[1]/som_width
	unit_height = image.shape[0]/som_height

	active_unit_matrix = np.zeros([som_width, som_height], dtype = int)

	for l in range(som_height):
		for c in range(som_width):
			x = int(c * unit_width)
			y = int(l * unit_height)

			active_unit_matrix[c, l] = (1 if is_active_unit(image, x, y, int(unit_width), int(unit_height)) else 0)

	return active_unit_matrix

def is_out_of_bounds(array, x, y):
	return x < 0 or y < 0 or array.shape[0] <= x or array.shape[1] <= y

def calculate_distances_for_unit(active_unit_matrix, distance_matrix, unit):
	visited_units = active_unit_matrix != 1
	active_units = []
	next_active_units = [unit]
	visited_units[unit[0], unit[1]] = True

	distance = 0 

	while len(next_active_units) > 0:
		active_units = next_active_units
		next_active_units = []

		while len(active_units) > 0:
			current_unit = active_units.pop(0)
			x = current_unit[0]
			y = current_unit[1]

			distance_matrix.set_distance_by_coords(unit[0], unit[1], x, y, distance)

			# neighbourhood
			if not is_out_of_bounds(visited_units, x-1, y) and not visited_units[x-1, y]:
				next_active_units.append([x-1, y])
				visited_units[x-1, y] = True
			if not is_out_of_bounds(visited_units, x+1, y) and not visited_units[x+1, y]:
				next_active_units.append([x+1, y])
				visited_units[x+1, y] = True
			if not is_out_of_bounds(visited_units, x, y-1) and not visited_units[x, y-1]:
				next_active_units.append([x, y-1])
				visited_units[x, y-1] = True
			if not is_out_of_bounds(visited_units, x, y+1) and not visited_units[x, y+1]:
				next_active_units.append([x, y+1])
				visited_units[x, y+1] = True

		#print(next_active_units)

		distance += 1

def calculate_distance_matrix(active_unit_matrix):
	distance_matrix = som_tools.DistanceMatrix(active_unit_matrix)

	for i, unit in enumerate(distance_matrix.get_active_units()):
		#print(i, len(distance_matrix.get_active_units()))
		calculate_distances_for_unit(active_unit_matrix, distance_matrix, unit)

	return distance_matrix

def load_mnemonic_image(image_path, som_width, som_height):
	img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	img = convert_to_binary(img)

	active_unit_matrix = convert_to_active_unit_matrix(img, som_width, som_height)
	distance_matrix = calculate_distance_matrix(active_unit_matrix)

	return active_unit_matrix, distance_matrix