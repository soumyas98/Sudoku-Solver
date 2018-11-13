import cv2

for i in range(0, 10):
    filename = '{}.png'.format(i)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(filename, img)
