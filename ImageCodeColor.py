import numpy as np
import matplotlib.pyplot as plt
import cv2
from ReedSolomon import ReedSolomon as RS

PIXELS_PER_BLOCK = 20
BLOCK_WIDTH = 2
BLOCK_HEIGHT = 2
ERROR_CORRECTION_BYTES = 16
SIZE = 16

# OpenCV uses BGR instead of RGB. Why? Who knows.
# Did this take me way too long to figure out? Yes.
# Will I ever remember this? Probably not.
COLOR_MAPPING = {
    0: (255, 255, 255), # White
    1: (0, 0, 0),       # Black
    2: (0, 0, 255),     # Red
    3: (0, 255, 0),     # Green
}

COLOR_MAPPING_LABELS = {
    "White": 0,
    "Black": 1,
    "Red": 2,
    "Green": 3,
}

class ImageCoder:
    def __init__(self, content="", size=SIZE, pixelsPerBlock=PIXELS_PER_BLOCK, blockWidth=BLOCK_WIDTH, blockHeight=BLOCK_HEIGHT, errorCorrectionBytes=ERROR_CORRECTION_BYTES):
        self.grid = np.zeros((size, size, 3), dtype=int)
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

    def convertBase10ToBaseN(self, val, base=4, width=4):
        """
        Convert a base 10 integer to a base N integer with a given width.
        Padding is done with leading zeros.
        Examples:
            5 ->  0011
            10 -> 0022
            200 -> 3020
        """
        if val == 0:
            return "0" * width
        digits = []
        while val > 0:
            val, remainder = val // base, val % base
            digits.append(str(remainder))
            
        while len(digits) < width:
            digits.append('0')
            
        return ''.join(digits[::-1])

    def generateBlock(self, val):
        width = self.blockWidth
        height = self.blockHeight

        base_str = self.convertBase10ToBaseN(val, base=4, width=width * height)
        index = 0

        # Convert integer to height x width x 3 block
        block = np.zeros((height, width, 3), dtype=int)
        for row in range(height):
            for column in range(width):
                block[row, column] = COLOR_MAPPING[int(base_str[index])]
                index += 1
        return block
    
    def generateCode(self):
        width = self.blockWidth
        height = self.blockHeight

        # If content an array, skip this step
        integers = None
        if (type(self.content) == str):
            # Convert letter to UTF-8 integer
            integers = [ord(letter) for letter in self.content]
        else:
            integers = self.content

        x, y = 0, 0
        # print("Integers Before Padding:", integers)

        # Last integer is the length of the message. After that, pad with 0s until the end of the grid with error correction bytes
        integers.append(len(self.content))
        while (len(integers) + self.errorCorrectionBytes < self.maximumAvailableBytes):
            integers.append(0)
        
        # print("Integers After Padding:", integers)
      
        # Add error correction bytes
        integers = RS.encode(integers, self.errorCorrectionBytes, intArray=True)
        
        # Segment grid into height x width blocks which will store the base 4 representation of the UTF-8 integer
        for integer in integers:
            block = self.generateBlock(integer)
            self.grid[y:y+height, x:x+width] = block
            x += width
            if (x >= self.size):
                x = 0
                y += height
      
    def generateBorder(self, img: np.ndarray, intensity=0):
        """
        Generates a border around the image. With alignment on the top left.
        """
        img_width = img.shape[1]
        img_height = img.shape[0]

        new_img = np.ones((img_height + 2 * self.pixelsPerBlock, img_width + 2 * self.pixelsPerBlock, 3), dtype=np.uint8) * intensity
        new_img[self.pixelsPerBlock:img_height + self.pixelsPerBlock, self.pixelsPerBlock:img_width + self.pixelsPerBlock] = img

        return new_img

    def generateBorders(self, img:np.ndarray):
        # Black border
        new_img = self.generateBorder(img, intensity=0)

        # White border
        new_img = self.generateBorder(new_img, intensity=255)

        # Black border
        new_img = self.generateBorder(new_img, intensity=0)

        return new_img

    def increaseImageSize(self, img:np.ndarray, factor:int):
        """
        Increases the size of the color image (n, n, 3) by a factor.
        """
        img_width = img.shape[1]
        img_height = img.shape[0]

        new_img = np.zeros((img_height * factor, img_width * factor, 3), dtype=np.uint8)
        for row in range(img_height):
            for column in range(img_width):
                new_img[row * factor:(row + 1) * factor, column * factor:(column + 1) * factor] = img[row, column]
        return new_img

    def generateImage(self, filename="image.png"):
        """
        Generates color image from the grid.
        """

        img = self.increaseImageSize(self.grid, factor=self.pixelsPerBlock)
        
        # print("Image Shape:", img.shape)
        
        # Generate border
        img = self.generateBorders(img)
        
        # Show the image
        # plt.imshow(img)
        # plt.show()

        # Save the image
        cv2.imwrite(filename, img)

        return img
        
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
        img[row - half_size:row + half_size, col - half_size:col + half_size] = (255, 0, 0)
    
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

    def convertBaseNtoBase10(self, val, base=4):
        """
        Convert a base N integer to a base 10 integer.
        Examples:
            0011 -> 5
            0022 -> 10
            3020 -> 200
        """
        val = str(val)
        base10 = 0
        for digit in val:
            base10 = base10 * base + int(digit)
        return base10

    def decodeBlock(self, block, base=4):
        """
        Decodes the block and returns the decoded data as an integer.
        Top left is most significant bit. Bottom right is least significant bit.
        """
        bits = []
        for row in range(block.shape[0]):
            for col in range(block.shape[1]):
                bits.append(block[row, col])
        
        # Convert bits to integer
        return self.convertBaseNtoBase10("".join([str(bit) for bit in bits]), base=base)

    
    def decodeDataArray(self, data, base=4):
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
        numBlocks = self.size + 2
        width = 20
        
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
                block = self.getCubeFromCenter(self.img, row, col, 5)
                median = np.median(block, axis=(0, 1))                
                # If all channels are above 128, then it is white
                if (median[0] > 128 and median[1] > 128 and median[2] > 128):
                    data[i, j] = COLOR_MAPPING_LABELS["White"]

                # If all channels are below 128, then it is black
                elif (median[0] < 128 and median[1] < 128 and median[2] < 128):
                    data[i, j] = COLOR_MAPPING_LABELS["Black"]

                # If red is above 128, then it is red
                elif (median[2] > 128):
                    data[i, j] = COLOR_MAPPING_LABELS["Red"]

                # If green is above 128, then it is Green
                elif (median[1] > 128):
                    data[i, j] = COLOR_MAPPING_LABELS["Green"]

                else:
                    data[i, j] = 0
                    print("ERROR: Unknown color!", median)
                
                # Use for report!
                # self.drawCubeFromCenter(self.img, row, col, 5)

                col += width
            col = int(1.5 * width)
            row += width
        
        # Show the image
        # cv2.imshow("Image", self.img)
        # cv2.waitKey(0)
        
        # Decode the data
        integerData = self.decodeDataArray(data)
        decodedData = RS.decode(integerData, self.errorCorrectionBytes)

        if (decodedData == None):
            print("ERROR: Could not decode data!")
            return ""
        
        # Start from end of decoded data and remove padding
        while (decodedData[-1] == 0):
            decodedData.pop()
        msgLength = decodedData.pop()

        # Convert to UTF-8 string
        decodedData = "".join(map(chr, decodedData))

        # print(data)
        return decodedData

        # # Show the image
        # plt.imshow(self.img)
        # plt.show()