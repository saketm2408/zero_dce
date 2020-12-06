from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
import sys


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
  image_name = sys.argv[1]

  interpreter = tflite.Interpreter(model_path="zdce.tflite")
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  floating_model = input_details[0]['dtype'] == np.float32

  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  im = cv2.resize(cv2.imread(image_name), (600, 400))
  input_data = np.expand_dims(im, axis=0)
  if floating_model:
    input_data = np.float32(input_data) / 255.0

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  output_data = output_data[0]
  plt.imsave("output.png",output_data)
