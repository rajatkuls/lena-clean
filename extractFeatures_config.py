# Put down all the paths here
import os

WAV_PATH = '../data/VanDam/'
DATA_PATH = '../data/'
OUTPUT_FEATURES_PATH = '../data/features/st/VanDam/'
os.system('mkdir -p '+DATA_PATH+'features/')

FOLDS_PATH = '../data/folds/VanDam/portion*'


FRAME_WIDTH = 0.001
INV_FRAME_WIDTH = 1000
MIN_DUR = 0.0001


stWin = 0.05
stStep = 0.05






