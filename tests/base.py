# Modify path to include the parent directory
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Include all project modules
from ImageCodeColor import ImageCoder as Code
from ImageCodeColor import ImageDecoder as Decode
from ImageCode import ImageDecoder as DecodeBW
from ImageCode import ImageCoder as CodeBW
from ImageDetector import *