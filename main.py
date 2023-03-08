from ImageCodeColor import ImageCoder as Code
from ImageCodeColor import ImageDecoder as Decode

code = Code("Hello World Color Version! I can fit much more")
code.generateImage(filename="helloworldcolor.png")

# decode = Decode("imgs/helloworld.png")