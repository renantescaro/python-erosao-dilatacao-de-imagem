import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('assets/brazilian_vehicle_license_plate.jpg')

# erosão
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
imagem = cv2.erode(imagem, rect, iterations=2)

# dilatação
elip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
imagem = cv2.dilate(imagem, elip, iterations=5)


plt.imsave('assets/final.png', imagem)
plt.imshow(imagem)
plt.show()
