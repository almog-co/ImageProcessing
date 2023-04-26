from base import *

TEXT = "Hello world color version! I can fit more data"

# Generate a color code
code = Code(TEXT)
code.generateImage(filename="helloworld_color_test.jpg")

# Decode the color code
img = detectCode("helloworld_green.jpg")
decode = Decode(img)

print(decode)

# if str(decode) == TEXT:
#     print("Color code test passed!")
# else:
#     print("Color code test failed!")