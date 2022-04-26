import cv2
import matplotlib.pyplot as plt
from PIL import Image
from binarizacao import Binarizacao


def salvar_imagem(caminho, imagem):
    plt.imsave(caminho, imagem)
    plt.imshow(imagem)
    plt.show()


resolucao_corte = (545, 180)
caminho_imagens = 'assets/carros/'
imagem = cv2.imread(f'{caminho_imagens}original.jpg')
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# redimensionamento
imagem = cv2.resize(imagem, resolucao_corte)
salvar_imagem(f'{caminho_imagens}/0.png', imagem)

# binarização
imagem = Binarizacao(
    imagem=Image.fromarray(imagem),
    limiar=93
).processar()
salvar_imagem(f'{caminho_imagens}/1.png', imagem)

# erosão
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
imagem = cv2.erode(imagem, rect, iterations=1)
salvar_imagem(f'{caminho_imagens}/2.png', imagem)

# dilatação
elip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
imagem = cv2.dilate(imagem, elip, iterations=2)
salvar_imagem(f'{caminho_imagens}/3.png', imagem)
