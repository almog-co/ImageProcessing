import argparse
import cv2
from ImageCodeColor import ImageDecoder as Decode
from ImageDetector import *

parser = argparse.ArgumentParser(description='Decode a color code.')

# Define the filename argument
parser.add_argument('filename', help='name of the file to process')

# Define the verbose argument
parser.add_argument('-v', '--verbose', action='store_true',
                    help='increase output verbosity')

args = parser.parse_args()
filename = args.filename
verbose = args.verbose

img = detectCode(filename)
if (img is None):
    print("No code found!")
    exit()

if (verbose):
    cv2.imshow("Cropped Code", img)
    cv2.waitKey(1)

decode = Decode(img)
if (str(decode) == ""):
    print("Unable to decode!")
else:
    print("Decoded message:")
    print(decode)

# Wait until the user presses a key to exit
cv2.waitKey(0)


