import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import matplotlib.pyplot as plt
import cv2

import os
import random
import pickle
import collections
from collections import defaultdict
from pprint import pprint
import math
import time
import random
import pickle as p

from graph import *
from model import *

if __name__ == '__main__':
    model_graph(np.zeros((1, 1080, 1920, 3)), np.zeros((1, 1080, 1920, 3)), np.zeros((2, 100, 100, 3)), np.zeros((2, 100, 100, 3)), 
                    pretrained='/home/ubuntu18/low_light_enhancement_models/exdark/zdce_5.pickle', get_pd=True, it=8)
