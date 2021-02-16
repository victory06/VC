#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:26:52 2020

@author: victor
"""

import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
#Ejercicio1
def leeimagen(filename, flagcolor):
    img = cv2.imread(filename, flagcolor)
    return img

#Ejercicio2
def pintaI(im,numfigure):
    if(im.min()<0 ):
        im=im-im.min()
    if(im.max()!=0):
        im = im / im.max()
    plt.figure(numfigure)
    if im.size==3:
        plt.imshow(im)
    else:
        plt.imshow(im,cmap='gray')
    return 0

#Ejercicio3
def pintaMI(vim, fignumber): 
    #calculamos altura mínima y mantenemos las proporciones
    h_min = min(img.shape[0] for img in vim)
    vim_resize = [cv2.resize(img, (int(img.shape[1]*h_min/img.shape[0]), h_min)) for img in vim]
    #Cambiamos el formato a color
    for i in range(len(vim)): 
        if len(vim[i].shape)!=3:
            vim_resize[i] = cv2.cvtColor(vim_resize[i].astype('uint8'), cv2.COLOR_GRAY2BGR)
    img_final=cv2.hconcat(vim_resize)
    plt.figure(fignumber)
    plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
    return 0

#Ejercicio4 de entrada una lista de pixeles, la imagen y el color al que cambiar
def cambiaColor(lista, img, color):
    for pix in lista:
        img[pix[1], pix[0]]=color
    return img

#Ejercicio5 de entrada una lista de imagenes y sus titulos
def pintaMIT(vim, titles, fignumber):
    if len(vim)==len(titles):
        h_min = min(im.shape[0] for im in vim)
        borde = [0.5,0.5,0.5]     #Color del borde
        vim = [cv2.resize(im, (int(im.shape[1]*h_min/im.shape[0]), h_min)) for im in vim]
        for i in range(len(vim)):
            if(len(vim[i].shape) != 3):
                vim[i] = cv2.cvtColor(vim[i].astype('uint8'), cv2.COLOR_GRAY2BGR) 
            vim[i]=cv2.copyMakeBorder(vim[i],100,10,10,10,cv2.BORDER_CONSTANT,value=borde )
            cv2.putText(vim[i], titles[i], (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255), 3, 0)
            vim[i]=cv2.cvtColor(vim[i], cv2.COLOR_BGR2RGB)
        vis = cv2.hconcat(vim)
        pintaI(vis, fignumber)
    else:
        print("No hay el mismo número de imagenes que de titulos")
    return 0


#Ejecucion Ej1
imgC=leeimagen('./images/logoOpenCV.jpg', -1)
imgG=leeimagen('./images/logoOpenCV.jpg', 0)
plt.figure(1)
plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB))
plt.figure(2)
plt.imshow(cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB))
plt.axis("off")

#Ejecucion Ej2
im = np.random.uniform(low=0, high=255, size=(291,450,3) )
pintaI(im,3)
im2 = np.random.uniform(low=-500, high=500, size=(291,450) )
pintaI(im2,4)

#Ejecución Ej3
im1 = cv2.imread('./images/logoOpenCV.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('./images/messi.jpg', cv2.IMREAD_GRAYSCALE)
im3 = cv2.imread('./images/orapple.jpg', -1)
vim=np.array([im1,im2,im3])
pintaMI(vim,5)

#Ejecución Ej4
lista=[[1,1], [4,3], [2,5], [23,25], [23,26], [23,27], [23,28], [23,29]]
color=[0,0,0]
cambiaColor(lista,im3,color)
plt.figure(6)
plt.imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB))

#Ejecucion Ej5
titulos=["logoOpenCV", "Messi", "orapple"]
pintaMIT(vim,titulos, 7)


