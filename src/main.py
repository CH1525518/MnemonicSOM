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

def som_test(initial_learning_rate = .1, learning_rate_decay = 10, initial_radius = 1, radius_decay = 10):
	active_unit_matrix, distance_matrix = input_utils.load_mnemonic_image("mnemonics/stick_figure.png", 25, 25)

	print("reading input")
	X, y = input_utils.read_dataset("datasets/", "chainlink")
	#X, y = input_utils.read_dataset("datasets/", "10clusters")
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
	som.train(X, 1)
	print("training done")

	#som.show_som(X, y)
	#som.show_som2(X)

	return som

def main():
	#mnemonics_test()
	X, y = input_utils.read_dataset("datasets/", "chainlink")

	# chainlink
	# learning rate 1-2
	# learning rate decay 5

	som = som_test(initial_learning_rate = 1.5, learning_rate_decay = 5, initial_radius = 1, radius_decay = 10)
	
	#plt.title('Learning rate decay = 5')
	som.show_som(X, y)

	som.export_weightlist()

if __name__ == "__main__":
	main()