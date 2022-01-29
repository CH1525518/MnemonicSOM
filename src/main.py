import numpy as np
import matplotlib.pyplot as plt

import input_utils
import som_tools

def mnemonics_test():
	img = cv2.imread("mnemonics/stick_figure.png", cv2.IMREAD_UNCHANGED)
	img = convert_to_binary(img)

	active_unit_matrix = convert_to_active_unit_matrix(img, 20, 20)

	distance_matrix = calculate_distance_matrix(active_unit_matrix)

	x = distance_matrix.get_active_units()[118][0]
	y = distance_matrix.get_active_units()[118][1]
	distance_matrix.show_distance_map(active_unit_matrix.shape[1], active_unit_matrix.shape[0], x, y)

	plt.imshow(img)
	plt.show()

	plt.imshow(np.swapaxes(active_unit_matrix, 0, 1))
	plt.show()

def som_test():
	active_unit_matrix, distance_matrix = input_utils.load_mnemonic_image("mnemonics/stick_figure.png", 20, 20)

	print("reading input")
	X, y = input_utils.read_dataset("datasets/", "chainlink")
	print("reading input done")

	print("generating som")
	som = som_tools.MnemonicSOM(active_unit_matrix, distance_matrix, len(X[0]), 
		initial_learning_rate = .1, 
		learning_rate_decay = 10, 
		initial_radius = 1, 
		radius_decay = 10)
	print("generating som done")

	som.show_som(X, y)
	som.show_som2(X)

	print("started training")
	som.train(X, 10)
	print("training done")

	som.show_som(X, y)
	som.show_som2(X)

def main():
	#mnemonics_test()
	som_test()

if __name__ == "__main__":
	main()