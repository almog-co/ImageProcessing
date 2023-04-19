import cv2 
import numpy as np
from matplotlib import pyplot as plt

def detectCode(filename):
    img = cv2.imread(filename)

    # Reduce resolution
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run canny to find edges then bounding box
    edges = cv2.Canny(gray, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours that are too small
    contours = [c for c in contours if cv2.contourArea(c) > 5000]

    # Filter out contours that are too close together
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    end = False
    while (not end):
        end = True
        for i in range(len(contours) - 1):
            if abs(cv2.contourArea(contours[i]) - cv2.contourArea(contours[i+1])) < 500:
                contours.pop(i)
                end = False
                break
            
    # Find the contours where there are 3 contours where area is increasing by at most 10% between contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    startIndex = 0
    for i in range(len(contours) - 3):
        if cv2.contourArea(contours[i]) > 0.9 * cv2.contourArea(contours[i+1]) and cv2.contourArea(contours[i+1]) > 0.9 * cv2.contourArea(contours[i+2]):
            startIndex = i
            break
    contours = contours[startIndex:startIndex+3]

    # For report!
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("Contours", img)
    # cv2.waitKey(0)

    # Get the smallest contour
    contours = sorted(contours, key=cv2.contourArea)

    # Get the corners of the innermost contour
    corners = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
    # print(corners)

    # Plot the corners. For report!
    for corner in corners:
        cv2.circle(img, (corner[0][0], corner[0][1]), 5, (0, 255, 0), -1)
    
    fixedCorners = []
    for corner in corners:
        # Format is (row, col)
        fixedCorners.append((corner[0][1], corner[0][0]))
    corners = fixedCorners

    # print(corners)
    
    # Find the order of the corners (top left, top right, bottom right, bottom left)
    corners = sorted(corners, key=lambda x: x[0])
    # print(corners)
    topleftCorner, toprightCorner, bottomrightCorner, bottomleftCorner = None, None, None, None
    if corners[0][1] < corners[1][1]:
        topleftCorner, toprightCorner = corners[0], corners[1]
    else:
        topleftCorner, toprightCorner = corners[1], corners[0]
    
    if (corners[2][1] < corners[3][1]):
        bottomleftCorner, bottomrightCorner = corners[2], corners[3]
    else:
        bottomleftCorner, bottomrightCorner = corners[3], corners[2]

    # print(topleftCorner, toprightCorner, bottomrightCorner, bottomleftCorner)

    # Solve system of linear equations to find arguments for projective transform
    originalCorners = np.matrix([
        [topleftCorner[0], toprightCorner[0], bottomleftCorner[0], bottomrightCorner[0]],
        [topleftCorner[1], toprightCorner[1], bottomleftCorner[1], bottomrightCorner[1]],
        [1, 1, 1, 1]
    ])

    newCorners = np.matrix('0 0 360 360; 0 360 0 360; 1 1 1 1')

    # Let A = [a b c; d e f; g h 1]
    # Let x = originalCorners
    # Let b = newCorners
    # Ax = b. Solve for A
    # Solve for A = B * x^T * (x * x^T)^-1
    x = originalCorners
    xT= x.transpose()
    b = newCorners
    A = b * xT * np.linalg.inv(x * xT)

    """
    A is a 3x3 matrix. Where transform is
    A * [x; y; 1] = [x'; y'; 1]
    """

    # [col, row] format because opencv is actually stupid
    beforePoints = [
        [topleftCorner[1], topleftCorner[0]],     # Top left
        [bottomleftCorner[1], bottomleftCorner[0]], # Bottom left
        [bottomrightCorner[1], bottomrightCorner[0]], # Bottom right
        [toprightCorner[1], toprightCorner[0]],   # Top right
    ]

    afterPoints = [
        [0, 0],     # Top left
        [0, 360],   # Bottom left
        [360, 360], # Bottom right
        [360, 0]    # Top right
    ]

    M = cv2.getPerspectiveTransform(np.float32(beforePoints), np.float32(afterPoints))

    # Apply transform to each pixel using imwarp with bilinear interpolation
    img = cv2.warpPerspective(img, M, (360, 360), flags=cv2.INTER_LINEAR)



    

    # Run hough transform to find horizontal lines
    # edges = cv2.Canny(gray, 100, 200)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 130)
    
    # # Visualize the lines - Use for report!
    # for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #     p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    #     cv2.line(img, p1, p2, (0, 0, 255), 2)

    # Show img
    #plt.imshow(edges, cmap = 'gray')
    #plt.figure()
    #plt.imshow(img, cmap = 'gray')
    #plt.show()
    return img
