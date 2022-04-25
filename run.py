import cv2
import matplotlib.pyplot as plt
from PIL import Image
from binarizacao import Binarizacao


imagem = cv2.imread('assets/original.jpg')
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# redimensionamento
imagem = cv2.resize(imagem, (600, 195))
plt.imsave('assets/0.png', imagem)
plt.imshow(imagem)
plt.show()

# binarização
imagem = Binarizacao(
    imagem=Image.fromarray(imagem),
    limiar=93
).processar()
plt.imsave('assets/1.png', imagem)
plt.imshow(imagem)
plt.show()

# erosão
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
imagem = cv2.erode(imagem, rect, iterations=1)
plt.imsave('assets/2.png', imagem)
plt.imshow(imagem)
plt.show()

# dilatação
elip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
imagem = cv2.dilate(imagem, elip, iterations=2)
plt.imsave('assets/3.png', imagem)
plt.imshow(imagem)
plt.show()
