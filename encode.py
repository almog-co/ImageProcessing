import argparse
import cv2
from ImageCodeColor import ImageCoder as Code

parser = argparse.ArgumentParser(description='Decode a color code.')

# Define the text argument
parser.add_argument('text', help='text to encode')

# Define the verbose argument
parser.add_argument('-v', '--verbose', action='store_true',
                    help='increase output verbosity')

# Define optional location to store output file. Default is 'code.png'
parser.add_argument('-o', '--output', default='code.png',
                    help='output file name')

args = parser.parse_args()
text = args.text
verbose = args.verbose
output = args.output

code = Code(text, verbose=verbose)
img = code.generateImage(filename=output)

if (verbose):
    cv2.imshow("Generated Code", img)
    cv2.waitKey(0)

