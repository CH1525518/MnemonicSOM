import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

import input_utils
import som_tools

def mnemonics_test():
	img = cv2.imread("mnemonics/stick_figure.png", cv2.IMREAD_UNCHANGED)
	img = input_utils.convert_to_binary(img)

	active_unit_matrix = input_utils.convert_to_active_unit_matrix(img, 50, 50)

	distance_matrix = input_utils.calculate_distance_matrix(active_unit_matrix)

	x = distance_matrix.get_active_units()[118][0]
	y = distance_matrix.get_active_units()[118][1]
	distance_matrix.show_distance_map(active_unit_matrix.shape[1], active_unit_matrix.shape[0], x, y)

	plt.imshow(img)
	plt.show()

	plt.imshow(np.swapaxes(active_unit_matrix, 0, 1))
	plt.show()

def som_test(input_image = 'stick_figure.png', input_dataset = "chainlink",
             som_width = 50, som_height = 50,
	     initial_learning_rate = .1, learning_rate_decay = 10, initial_radius = 1, radius_decay = 10, 
             initial_radius = 1, radius_decay = 10, n_epochs = 1):
	image_path = '../mnemonics/'
	active_unit_matrix, distance_matrix = input_utils.load_mnemonic_image(image_path+input_image, som_width, som_height)

	print("reading input")
	#X, y = input_utils.read_dataset("datasets/", "chainlink")
	X, y = input_utils.read_dataset("../datasets/", input_dataset)
	print("reading input done")

	print("generating som")
	som = som_tools.MnemonicSOM(active_unit_matrix, distance_matrix, len(X[0]), 
		initial_learning_rate = initial_learning_rate, 
		learning_rate_decay = learning_rate_decay, 
		initial_radius = initial_radius, 
		radius_decay = radius_decay)
	print("generating som done")

	#som.show_som(X, y)
	#som.show_som2(X)

	print("started training")
	som.train(X, 10)
	print("training done")

	#som.show_som(X, y)
	#som.show_som2(X)

	return som

def main():
	#mnemonics_test()
	#X, y = input_utils.read_dataset("datasets/", "chainlink")
	X, y = input_utils.read_dataset("datasets/", "10clusters")

	# chainlink
	# learning rate 1-2
	# learning rate decay 5
	# initial radius 1-2
	# initial radius 10

	# 10clusters
	# learning rate 2
	# learning rate decay 5
	# initial radius 2
	# initial radius 10

	# Note: with more active units (larger size of the unitmatrix), the possabilty for holes in the visual representation of the network increases
	# due to the units being colored according to their fireing (this could be fixed using a different coloring technique but since this is not
	# part of the assignment, there is no need to do that). This effect however has lead to the realization that the radius_decay greatly influences
	# the shape of the SOM in terms of the size of the clusters. Larger radius decay leads to larger clusters while smaller radius decay leads to small
	# clusters with most of the units not being used.

	som = som_test(initial_learning_rate = 2.5, learning_rate_decay = 5, initial_radius = 2, radius_decay = 10)
	plt.title('radius_decay = 2.5')
	som.show_som(X, y)

	#som.export_weightlist()

if __name__ == "__main__":
	main()
