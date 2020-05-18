
import argparse
from tensorflow.keras.applications import vgg16
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--save_file_path', default='data/imagenet_probs.dat', type=str,
                    help='Name of file to save probs, labels pair.')
parser.add_argument('--load_folder', default='.', type=str,
                    help='Path to folder containing ImageNet images.')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size of input data to model.')

if __name__ == "__main__":
	args = parser.parse_args()
	vgg_model = vgg16.VGG16(weights='imagenet')
	eval_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
	generator = eval_datagen.flow_from_directory(
		args.load_folder, shuffle=False, batch_size=args.batch_size, target_size=(224, 224))
	labels = generator.classes
	num_steps = int(np.ceil(len(labels) * 1.0 / args.batch_size))
	probs = vgg_model.predict_generator(generator, steps=num_steps)
	pickle.dump((probs, labels), open(args.save_file_path, "wb"))
