import numpy as np
import matplotlib.pyplot as plt
import cv2
from ReedSolomon import ReedSolomon as RS

PIXELS_PER_BLOCK = 20
BLOCK_WIDTH = 4
BLOCK_HEIGHT = 2
ERROR_CORRECTION_BYTES = 16
SIZE = 16

COLOR_MAPPING = {
    "R": (255, 0, 0),
    "L": (0, 0, 255),
    "B": (0, 0, 0),
    "W": (255, 255, 255),
}

class ImageCoder:
    def __init__(self, content="", size=SIZE, pixelsPerBlock=PIXELS_PER_BLOCK, blockWidth=BLOCK_WIDTH, blockHeight=BLOCK_HEIGHT, errorCorrectionBytes=ERROR_CORRECTION_BYTES):
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

        integers = []
        x, y = 0, 0

        # If content an array, skip this step
        if (type(self.content) == str):
            # Convert letter to UTF-8 integer
            integers = [ord(letter) for letter in self.content]
        else:
            integers = self.content
            
        # print("Integers Before Padding:", integers)

        # Last integer is the length of the message. After that, pad with 0s until the end of the grid with error correction bytes
        integers.append(len(self.content))
        while (len(integers) + self.errorCorrectionBytes < self.maximumAvailableBytes):
            integers.append(0)
        
        # print("Integers After Padding:", integers)
      
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
            
        # print("Image Shape:", img.shape)
        
        # Flip black and white
        img = 255 - img

        # Generate border
        img = self.generateBorders(img)
        
        # Show the image
        # plt.imshow(img, cmap='gray')
        # plt.show()

        # Make the image color
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Save the image
        cv2.imwrite(filename, img)
        
class ImageDecoder:
    def __init__(self, imgFile, size=SIZE, pixelsPerBlock=PIXELS_PER_BLOCK, blockWidth=BLOCK_WIDTH, blockHeight=BLOCK_HEIGHT, errorCorrectionBytes=ERROR_CORRECTION_BYTES):
        img = imgFile

        # Check if image is valid
        if (img is None):
            print("Image is invalid")
            exit(1)

        self.img = img
        self.size = size
        self.pixelsPerBlock = pixelsPerBlock
        self.blockWidth = blockWidth
        self.blockHeight = blockHeight
        self.errorCorrectionBytes = errorCorrectionBytes

        # Parse image from left to right
        self.grid = np.zeros((size, size), dtype=int)
        self.decodedData = self.parseImage()
    
    def __str__(self):
        return self.decodedData
    
    def drawCubeFromCenter(self, img, row, col, size):
        """
        Draws bright green cube from the center of the point (row, col) with size=size.
        """
        half_size = size // 2
        img[row - half_size:row + half_size, col - half_size:col + half_size] = (0, 255, 0)
    
    def getCubeFromCenter(self, img, row, col, size):
        """
        Returns a cube from the center of the point (row, col) with size=size.
        """
        half_size = size // 2
        return img[row - half_size:row + half_size, col - half_size:col + half_size]

    def getRectangleFromTopLeft(self, img, row, col, width, height):
        """
        Returns a rectangle from the top left of the point (row, col) with width and height.
        """
        return img[row:row + height, col:col + width]

    def decodeBlock(self, block):
        """
        Decodes the block and returns the decoded data as an integer.
        Top left is most significant bit. Bottom right is least significant bit.
        """
        bits = []
        for row in range(block.shape[0]):
            for col in range(block.shape[1]):
                bits.append(block[row, col])
        
        # Convert bits to integer
        return int("".join(map(str, bits)), 2)

    
    def decodeDataArray(self, data):
        """
        Decodes the 2D byte data array and returns the decoded data as an array of integers.
        """
        b = []
        for row in range(0, data.shape[0], self.blockHeight):
            for col in range(0, data.shape[1], self.blockWidth):
                block = self.getRectangleFromTopLeft(data, row, col, self.blockWidth, self.blockHeight)
                b.append(self.decodeBlock(block))
        return b

    def parseImage(self):
        """
        Parses the image from left to right and top to bottom.
        """
        # img_width = self.img.shape[1]
        # img_height = self.img.shape[0]

        # # Run edge detection 
        # edges = cv2.Canny(self.img, 100, 200)

        # # Find contours
        # contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # # Sort contours by area. Find index where there is a big jump in area
        # delta = 0
        # boundingCountorIndex = None
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # for i in range(len(contours) - 1):
        #     if (cv2.contourArea(contours[i]) == 0):
        #         continue
        #     delta = (cv2.contourArea(contours[i]) - cv2.contourArea(contours[i + 1])) / cv2.contourArea(contours[i])
        #     if (delta > 0.5):
        #         boundingCountorIndex = i
        #         break
        
        # # Get the bounding box of the contour
        # x, y, w, h = cv2.boundingRect(contours[boundingCountorIndex])

        # # Use for report!
        # # print(len(contours))
        # # cv2.drawContours(self.img, contours, -1, (0, 255, 0), 2)

        # # Crop the image
        # self.img = self.img[y:y + h, x:x + w]

        # Get size of each block
        edges = cv2.Canny(self.img, 100, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if abs(cv2.boundingRect(contour)[2] - cv2.boundingRect(contour)[3]) < 2]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        medianContour = contours[len(contours) // 2]
        x, y, w, h = cv2.boundingRect(medianContour)
        width = w - 1
        # numBlocks = self.img.shape[0] // width
        width = self.img.shape[1] // (self.size + 2)
        numBlocks = self.size + 2
        
        # Visualize the median contour - USE FOR REPORT
        # cv2.drawContours(self.img, [medianContour], -1, (0, 255, 0), 2)

        # print("Block size:", width)
        # print("Number of blocks:", numBlocks)

        # Starts at 1 to skip the border. Go to middle of each block
        data = np.zeros((numBlocks - 2, numBlocks - 2), dtype=int)
        row, col = int(1.5 * width), int(1.5 * width)
        for i in range(numBlocks - 2):
            for j in range(numBlocks - 2):
                # Get median color of the 5x5 block from (row, col)
                median = np.median(self.img[row - 2:row + 2, col - 2:col + 2])
                data[i, j] = 0 if median > 127 else 1
                
                # Use for report!
                # self.drawCubeFromCenter(self.img, row, col, 5)

                col += width
            col = int(1.5 * width)
            row += width
        
        # Decode the data
        integerData = self.decodeDataArray(data)
        decodedData = RS.decode(integerData, self.errorCorrectionBytes)
        
        # Start from end of decoded data and remove padding
        while (decodedData[-1] == 0):
            decodedData.pop()
        msgLength = decodedData.pop()

        # Convert to UTF-8 string
        decodedData = "".join(map(chr, decodedData))

        # print(data)
        return decodedData

        # Show the image
        # plt.imshow(self.img)
        # plt.show()