from base import *
import os

TEXT = "Hello World!!!!"

# Generate a black and white code
code = CodeBW(TEXT)
code.generateImage(filename="helloworld_bw_test.png")

# Decode the black and white code
img = detectCode("helloworld_bw_test.png")
decode = DecodeBW(img)

if str(decode) == TEXT:
    print("Black and white code test passed!")
else:
    print("Black and white code test failed!")

# Delete the generated image
os.remove("helloworld_bw_test.png")

