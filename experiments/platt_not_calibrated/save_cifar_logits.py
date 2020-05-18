
import argparse
from tensorflow.keras.datasets import cifar10

import cifar10vgg
import lib.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--save_file_path', default='cifar_probs.dat', type=str,
                    help='Name of file to save probs, labels pair.')

if __name__ == "__main__":
	args = parser.parse_args()
	utils.save_test_probs_labels(cifar10, cifar10vgg.cifar10vgg(), args.save_file_path)
