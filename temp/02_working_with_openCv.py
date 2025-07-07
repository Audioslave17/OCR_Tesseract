import cv2
from matplotlib import pyplot as plt

image_file = "data/page_01.jpg"
image = cv2.imread(image_file)
# cv2.imshow("title",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    ax.imshow(im_data, cmap='gray')

    plt.show()

#display(image_file)

inverted_image = cv2.bitwise_not(image)
cv2.imwrite("temp/inverted_page_01.jpg", inverted_image)

#display("temp/inverted_page_01.jpg")

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = grayscale(image)
cv2.imwrite("temp/gray_img.jpg", gray_image)
#display("temp/gray_img.jpg")

thresh, im_bw = cv2.threshold(gray_image, 200, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_image.jpg", im_bw)
#display("temp/bw_image.jpg")


def noise_removal(img):
    import numpy as np
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    kernal = np.ones((1,1),np.uint8)
    img = cv2.erode(img, kernal, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.medianBlur(img,3)
    return img
no_noise = noise_removal(im_bw)
cv2.imwrite("temp/no_noise.jpg",no_noise)
#display("temp/no_noise.jpg")


def thick_font(img):
    import numpy as np
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations=1)
    img = cv2.bitwise_not(img)
    return img

dialated_image = thick_font(no_noise)
cv2.imwrite("temp/dialated_image.jpg",dialated_image)
#display("temp/dialated_image.jpg")

display("temp/no_noise.jpg")