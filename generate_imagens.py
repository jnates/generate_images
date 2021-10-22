import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy


from PIL import Image
import random
import os
from PIL import Image  
from scipy import ndimage
  

def load_images_from_folder(folder): 
  images = [] 
  for filename in os.listdir(folder): 
    img = Image.open(os.path.join(folder,filename)) 
    if img is not None: 
      images.append(img) 
  return images 


# Aumento de datos
def gausian_noise(img):  
  img = np.array(img)
  gaussian_noise=iaa.AdditiveGaussianNoise(10,40)
  imgn=gaussian_noise.augment_image(img)
  imgn = Image.fromarray(np.uint8(imgn)).convert('RGB')
  return imgn

def crop(img):
  img = numpy.array(img)
  crop = iaa.Crop(percent=(0, 0.3))
  imgn=crop.augment_image(img)
  imgn = Image.fromarray(np.uint8(imgn)).convert('RGB')
  return imgn

def flip(img):
  img = numpy.array(img) 
  flip_hr=iaa.Fliplr(p=1.0)
  imgn= flip_hr.augment_image(img)
  imgn = Image.fromarray(np.uint8(imgn)).convert('RGB')
  return imgn

def flipup(img):
  img = numpy.array(img)
  flip_vr=iaa.Flipud(p=1.0)
  imgn= flip_vr.augment_image(img)
  imgn = Image.fromarray(np.uint8(imgn)).convert('RGB')
  return imgn

def bright(img):
  img = np.array(img)
  gama = random.uniform(0.5, 1.0)
  contrast=iaa.GammaContrast(gamma=gama)
  imgn =contrast.augment_image(img)
  imgn = Image.fromarray(np.uint8(imgn)).convert('RGB')
  return imgn

################################################################

def data_aug(imag):
  imgn = imag
  a = random.choice([0,1,2])
  if a == 0:
    imgn = gausian_noise(imag)

  a = random.choice([0,1])
  if a == 0:
    imgn = crop(imgn)
    
  a = random.choice([0,1])
  if a == 0:
    imgn = flip(imgn)

  a = random.choice([0,1])
  if a == 0:
    imgn = flipup(imgn)

  imgn = bright(imgn)

  return imgn


def aumento_data(folder_in, folder_out, dimencion, n_new_images):
  for filename in os.listdir(folder_in):
    img = Image.open(os.path.join(folder_in,filename)).convert('RGB')
    new_img = img.resize((dimencion, dimencion))
    new_img.save(os.path.join(folder_out,filename),"jpeg")

    for i in range(0, n_new_images):
      new_img = data_aug(img)
      new_img = new_img.resize((dimencion, dimencion))
      new_img.save(os.path.join(folder_out, str(i) + filename),"jpeg")

      
aumento_data("/gdrive/My Drive/Colab/data/entrenamiento/buenas",
            "/gdrive/My Drive/Colab/data/aumentadas_buenas",
            224, 10)

from sklearn.model_selection import train_test_split 

buenas = load_images_from_folder("/gdrive/My Drive/Colab/data/aumentadas_buenas")
malas = load_images_from_folder("/gdrive/My Drive/Colab/data/aumentadas_malas")

train_buenas, test_buenas = train_test_split(buenas, test_size = 0.30)

train_malas, test_malas = train_test_split(malas, test_size = 0.30)


len(train_buenas)

n = 0
for elemento in train_buenas:
  n = n + 1
  elemento.save(os.path.join("/gdrive/My Drive/Colab/data/finales/entrenamiento/buenas",str(n)),'jpeg')


for elemento in test_buenas:
  n = n + 1
  elemento.save(os.path.join("/gdrive/My Drive/Colab/data/finales/validacion/buenas",str(n)),'jpeg')

n = 0
for elemento in train_malas:
  n = n + 1
  elemento.save(os.path.join("/gdrive/My Drive/Colab/data/finales/entrenamiento/malas",str(n)),'jpeg')
for elemento in test_malas:
  n = n + 1
  elemento.save(os.path.join("/gdrive/My Drive/Colab/data/finales/validacion/malas",str(n)),'jpeg')

for elemento in test_malas:
    n = n + 1
    elemento.save(os.path.join("/gdrive/My Drive/Colab/data/finales/validacion/malas",str(n)),'jpeg')

