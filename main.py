from ImageCodeColor import ImageCoder as Code
from ImageCodeColor import ImageDecoder as Decode
from ImageCode import ImageDecoder as DecodeBW
from ImageCode import ImageCoder as CodeBW
from ImageDetector import *
from matplotlib import pyplot as plt

# code = Code("Hello World Color Version! I can fit much more")
# code.generateImage(filename="helloworldcolor.png")

# decode = Decode("imgs/helloworldcolorerror.png")

# img = detectCode("imgs/helloworldside.png")
# decode = DecodeBW(img)
# print(decode)

# code = CodeBW("Hello World!!!")
# code = Code("Hello World Color Version! I can fit much more")
# code.generateImage()

img = detectCode("imgs/helloworldside3.jpg")
decode = DecodeBW(img)
print(decode)