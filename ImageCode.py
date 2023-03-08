import numpy as np
import matplotlib.pyplot as plt
from ReedSolomon import ReedSolomon as RS

class ImageCoder:
    def __init__(self, content="", size=16, pixelsPerBlock=20, blockWidth=4, blockHeight=2, errorCorrectionBytes=16):
        self.grid = np.zeros((size, size), dtype=int)
        self.pixelsPerBlock = pixelsPerBlock
        self.size = size
        self.content = content
        self.blockWidth = blockWidth
        self.blockHeight = blockHeight
        self.errorCorrectionBytes = errorCorrectionBytes
        self.maximumAvailableBytes= (size * size) // (blockWidth * blockHeight)
        self.maximumMessageLength = self.maximumAvailableBytes - self.errorCorrectionBytes - 1
        
        if (len(content) > self.maximumMessageLength):
            print("Content is too long for the grid size! Must be less than", self.maximumMessageLength, "bytes.")
            exit(1)

        self.generateCode()
    
    def generateBinaryFromInt(self, val, width=8):
        """
        Examples:
            5 ->  00000101
            10 -> 00001010
            15 -> 00001111
        """
        binary_str = bin(val)[2:]  # Remove the '0b' prefix from the binary string
        binary_str = binary_str.zfill(width)  # Pad the string with leading zeros
        return binary_str


    def generateBlock(self, val):
        width = self.blockWidth
        height = self.blockHeight

        binary_str = self.generateBinaryFromInt(val, width * height)
        index = 0

        # Convert integer to height x width block
        block = np.zeros((height, width), dtype=int)
        for row in range(height):
            for column in range(width):
                block[row, column] = binary_str[index]
                index += 1
        return block
        
    
    def generateCode(self):
        width = self.blockWidth
        height = self.blockHeight
    
        # Convert letter to UTF-8 integer
        x, y = 0, 0
        integers = [ord(letter) for letter in self.content]
        print("Integers Before Padding:", integers)

        # Last integer is the length of the message. After that, pad with 0s until the end of the grid with error correction bytes
        integers.append(len(self.content))
        while (len(integers) + self.errorCorrectionBytes < self.maximumAvailableBytes):
            integers.append(0)
        
        print("Integers After Padding:", integers)
      
        # Add error correction bytes
        integers = RS.encode(integers, self.errorCorrectionBytes, intArray=True)
        
        # Segment grid into height x width blocks which will store the binary representation of the UTF-8 integer
        for integer in integers:
            block = self.generateBlock(integer)
            self.grid[y:y+height, x:x+width] = block
            x += width
            if (x >= self.size):
                x = 0
                y += height
                
    def __str__(self):
        return self.generateGridLines()

    
    def generateGridLines(self):
        """
        Returns a string representation of the grid of size=8, width=4, height=2 separated by a line.
        -------------------
        0 0 0 0 | 0 0 0 0 |
        0 0 0 0 | 0 0 0 0 |
        -------------------
        0 0 0 0 | 0 0 0 0 |
        0 0 0 0 | 0 0 0 0 |
        -------------------
        0 0 0 0 | 0 0 0 0 |
        0 0 0 0 | 0 0 0 0 |
        -------------------
        """

        width = self.blockWidth
        height = self.blockHeight
        
        # Create the horizontal lines
        horiz_line = ' -' * (self.size - 1) + '\n'
        out = horiz_line

        # Create the vertical lines and values
        for row in range(self.size):
            for column in range(self.size):
                if (column % width == 0):
                    out += " | "
                out += str(self.grid[row, column])
            out += " |\n"
            if (row % height == height - 1):
                out += horiz_line
        
        return out

    def generateBorder(self, img: np.ndarray, intensity=0, alternate=False):
        """
        Generates a border around the image. With alignment on the top left.
        """
        img_width = img.shape[1]
        img_height = img.shape[0]

        new_img = np.ones((img_height + 2 * self.pixelsPerBlock, img_width + 2 * self.pixelsPerBlock), dtype=np.uint8) * intensity
        new_img[self.pixelsPerBlock:img_height + self.pixelsPerBlock, self.pixelsPerBlock:img_width + self.pixelsPerBlock] = img

        if (alternate):
            for i in range(img_width // self.pixelsPerBlock):
                if i % 2 == 0 and i != 0 and i != self.pixelsPerBlock - 1:
                    new_img[0:self.pixelsPerBlock, i * self.pixelsPerBlock: (i + 1) * self.pixelsPerBlock] = 255 - intensity
        
        return new_img

    def generateBorders(self, img:np.ndarray):
        # Black border
        new_img = self.generateBorder(img, intensity=0)

        # White border
        new_img = self.generateBorder(new_img, intensity=255)

        # Black border
        new_img = self.generateBorder(new_img, intensity=0)

        return new_img

    def generateImage(self, filename="image.png"):
        """
        Generates black and white image from the grid.
        """

        img_width = self.size * self.pixelsPerBlock
        img_height = self.size * self.pixelsPerBlock

        # Create a new image from numpy
        img = np.zeros((img_height, img_width), dtype=np.uint8)
        for row in range(img_height):
            for column in range(img_width):
                img[row, column] = self.grid[row // self.pixelsPerBlock, column // self.pixelsPerBlock] * 255
            
        print("Image Shape:", img.shape)
        
        # Flip black and white
        img = 255 - img

        # Generate border
        img = self.generateBorders(img)
        
        # Show the image
        plt.imshow(img, cmap='gray')
        plt.show()
        

        

    
