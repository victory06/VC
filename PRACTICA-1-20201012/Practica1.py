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

#Función para mostrar imágenes
def pintaIm(im,name):
    im=normMatrix(im)
    cv2.imshow(name,im)
    cv2.waitKey(20)
    return im

#Función para normalizar matrices
def normMatrix(im):
    if(im.min()<0 ):
        im=im-im.min()
    if(im.max()!=0):
        im = im / im.max()
    return im

#Función gaussiana
def gaussian(sigma,mask):
    return np.exp(- 0.5 / (sigma * sigma) * mask ** 2)

#Primera derivada de la función gaussiana
def derGaussian(sigma,mask):
    return -(np.exp(- 0.5 / (sigma * sigma) * mask ** 2)*mask/(sigma*sigma))

#Segunda derivada de la función gaussiana
def secondDerGaussian(sigma,mask):
    return -((-mask**2 +sigma*sigma)/((sigma**4)*np.exp( 0.5 / (sigma * sigma) * mask ** 2)))

""" Esta función devuelve la máscara 1D gaussiana discretizando en el intervalo
[-3*sigma, 3*sigma]  con 2*[3*sigma]+1 elementos tal y como se explica en teoría.
Se aplica la función gaussiana a cada elemento del intervalo y se normaliza."""
def gaussianMask1D(sigma="foo", r="foo"):
    if(sigma!="foo"):
        #El mayor entero que no supere a 3*sigma será nuestro radio del intervalo
        r=math.floor(3 * sigma)
    if(r!="foo"):
        sigma=math.floor((r-1)/2)
    #Tomamos los elementos en el intervalo de radio r
    mask=np.arange(-r, r+1)
    #Ignorando el coeficiente, aplicamos la gaussiana
    mask=gaussian(sigma,mask)
    #Normalizar
    mask=mask/mask.sum()       
    #Devolvemos la máscara
    return mask

"""La siguiente función deriva la máscara gaussiana tal y como se vio en teoría:
    para cada punto se resta la gaussiana en ese punto y en el siguiente y se divide
    entre 1. Si se quiere volver a derivar se hace de nuevo con la máscara derivada."""
def derivGaussianMask1D(nderiv, sigma="foo", r="foo"):
    if(sigma!="foo"):
        #El mayor entero que no supere a 3*sigma será nuestro radio del intervalo
        r=math.floor(3 * sigma)
    if(r!="foo"):
        sigma=math.floor((r-1)/2)
    #Tomamos los elementos en el intervalo de radio r
    mask=np.arange(-r, r+1)
    #Ignorando el coeficiente, aplicamos la derivada de gaussiana
    mask=derGaussian(sigma,mask)
    #Si se pide la segunda derivada, se repite el proceso
    if(nderiv==2):
        #Tomamos los elementos en el intervalo de radio r
        mask=np.arange(-r, r+1)
        #Ignorando el coeficiente, aplicamos la derivada de gaussiana
        mask=secondDerGaussian(sigma,mask)
    return mask

"""Función para aplicar la máscara gaussiana a un vector 1D llamado fila.
   Para ello, rellenamos de 0 o reflejo por fuera de los bordes con k-1 puntos para
   poder aplicar la máscara."""
def apply1DMask(fila, mask, sigma="foo", k="foo", modo_borde="cero"):
    #Para convolucionar, damos la vuelta a la máscara y la volvemos 1D
    mask = np.flip(mask.flatten())
    #Segunda matriz para introducir la imagen img con la máscara aplicada
    fila2=np.zeros(fila.shape)
    if(sigma!="foo"):
        #k será el rango, que será el resultado de la siguiente operación:
        k=int(math.floor(3*sigma))
    if(modo_borde=="cero"):
        #Relleno de 0 por fuera de la imagen para poder aplicar la máscara
        fila_rell=[0 for x in range(0,k)]
        fila2=np.concatenate((fila_rell,fila))
        fila2=np.concatenate((fila2,fila_rell))
    elif(modo_borde=="reflect"):
        #bordes reflejados
        fila2=np.pad(fila,(k,k),mode='reflect')
    else:
        #bordes replicados
        fila2=np.pad(fila,(k,k),mode='edge')
    #Esta fila será la solución y la que devolvamos
    fila_ret=[]
    #Con estos bucles aplicamos la máscara a la fila que hemos rellenado con 0
    for i in range(k,len(fila)+k):
        valor=0
        #Aplicamos máscara
        for j in range (-k,k+1):
            valor= valor + fila2[i+j] * mask[j+k]    
        #Introducimos en la fila resultado
        fila_ret.append(valor)        
    return fila_ret

"""Esta función calcula la convolución de una imágen con una matriz separable,
de ahí que tenga dos matrices 1D de entrada, una para aplicar a las filas y otra
para aplicar a las columnas"""
def applySeparable2DMask(img, maskx, masky, modo_borde="cero"):
    #Pasamos a float
    img = img.astype(float)
    #Caso de imagen en color, tenemos en cuenta los canales de color
    if(len(img.shape)==3):
        #Tomamos las dimensiones de la imágen: filas columnas y canales de color
        F, C, CC=img.shape
        #Cantidad de relleno en los bordes de la imagen la imagen
        relleno=math.floor((maskx.shape[0] - 1) / 2.0)
        # Por cada canal de color aplicamos la convolución
        for k in range(CC):
            if(modo_borde=="cero"):
                #Canal de color donde rellenamos los bordes con 0
                canal=np.pad(img[:, :, k], (relleno, relleno), mode='constant', constant_values=0)
            elif(modo_borde=="reflect"):
               #Canal de color donde rellenamos con el modo reflejo
               canal=np.pad(img[:, :, k], (relleno, relleno), mode='reflect')
            else:
                #Bordes replicados
                canal=np.pad(img[:, :, k], (relleno, relleno), mode='edge')
            #Aplicamos en las columnas
            for j in range(C+2*relleno):
                #Convolucionamos la máscara a la columna j
                canal[:, j]=apply1DMask(canal[:, j], masky,k=relleno,modo_borde=modo_borde)
            #Aplicamos la primera máscara en las filas
            for i in range(F+2*relleno):
                #Convolucionamos la máscara a la fila i
                canal[i, :]=apply1DMask(canal[i, :], maskx,k=relleno,modo_borde=modo_borde)
            #Actualizamos la imagen sin el relleno
            img[:, :, k]=canal[relleno:(F+relleno), relleno:(C+relleno)]
    #Caso imagen en escala de grises
    if(len(img.shape)==2):
        #Tomamos las dimensiones de la imágen: filas y columnas solo pues es escala de grises
        F, C=img.shape
        #Cantidad de relleno en la imagen
        relleno=round((maskx.shape[0] - 1) / 2.0)
        if(modo_borde=="cero"):
            #Imagen auxiliar para rellenar los bordes con 0
            img_aux=np.pad(img[:, :], (relleno, relleno), mode='constant', constant_values=0)
        elif(modo_borde=="reflect"):
            #Imagen auxiliar para rellenar los bordes reflejados
            img_aux=np.pad(img[:, :], (relleno, relleno), mode='reflect')
        else:
            #Bordes Replicados
            img_aux=np.pad(img[:, :], (relleno, relleno), mode='edge')
        #Aplicamos en las columnas
        for j in range(C+2*relleno):
             #Convolucionamos la máscara a la columna j
            img_aux[:, j]=apply1DMask(img_aux[:, j], masky,k=relleno, modo_borde=modo_borde)
        #Aplicamos la primera máscara en las filas
        for i in range(F+2*relleno):
            #Convolucionamos la máscara a la fila i
            img_aux[i, :]=apply1DMask(img_aux[i, :], maskx,k=relleno,modo_borde=modo_borde)
        #Actualizamos la imagen sin el relleno
        img[:, :]= img_aux[relleno:(F+relleno), relleno:(C+relleno)]
    return img


"""Esta función será para devolver las máscaras de getDerivKernel y una imagen
sobre la que se han aplicado para hacer las comparaciones que pide el apartado C"""
def aplicaDerivKernel(img,dx, dy, ksize):
    kx,ky=cv2.getDerivKernels(dx,dy,ksize,normalize=True)    
    return kx,ky,applySeparable2DMask(img,kx,ky)

np.random.seed(1)

"""Aplica a la imagen img la laplaciana que consiste en convolucionar las máscaras
de las segundas derivadas con la imagen y sumarlas."""
def laplaciana(img, sigma, modo_borde="cero"):
    #Máscaras sin derivar
    maskx=gaussianMask1D(sigma)
    masky=gaussianMask1D(sigma)
    #Máscaras con la doble derivada
    maskx2=derivGaussianMask1D(2,sigma)
    masky2=derivGaussianMask1D(2,sigma)
    #Convolucionamoscon cada derivada con la imagen
    imgx = applySeparable2DMask(img, maskx2, masky, modo_borde=modo_borde)
    imgy = applySeparable2DMask(img, maskx, masky2, modo_borde=modo_borde)
    #Devolvemos la suma de las imagenes convolucionadas y multiplicamos
    #por sigma cuadrado para obtener la laplaciana normalizada
    #También devolvemos la máscara
    maskx2=np.matrix(maskx2)
    masky2=np.matrix(masky2)
    maskx=np.matrix(maskx)
    masky=np.matrix(masky)
    mask1=np.dot(maskx2.T,masky)
    mask2=np.dot(maskx.T,masky2)
    return (sigma*sigma)*(imgx + imgy), (sigma*sigma)*(mask1+mask2)


"""Esta función quita las filas y las columnas de ratiof en ratiof 
consiguiendo así quitar columnas y filas impares (downsampling)"""
def downsampling(img, ratio=2):
    img=np.array(img)
    return img[::ratio, ::ratio]

"""Añade filas y columnas duplicadas según el ratio indicado"""
def upsampling(im, ratio=2):
    #Copiamos para no cambiar la imagen original
    img=np.copy(im)
    #Número necesario de filas y columnas totales para la imagen upsampled
    numf=ratio*img.shape[0]
    numc=ratio*img.shape[1]
    #Primero repetimos las filas
    img2=np.repeat(img,ratio,axis=0).reshape(numf,img.shape[1])
    #Ahora repetimos las columnas y obtenemos la imagen upsampled
    imgres=np.repeat(img2,ratio).reshape(numf,numc)
    return imgres

"""Para crear la pirámide se sigue el siguiente método: se parte de la imagen
original que estará en el vector, despues se aplica el filtro y se quitan
filas y columnas impares (downsampling) luego se vuelve a aplicar el filtro
gaussiano y se vulve a hacer el downsampling, repitiendo esto hasta el
número de niveles nv."""
def piramideGaussiana(img, sigma, nv, modo_borde="reflect", ratiods=2):
    #Vector en el que guardamos las imágenes
    p=[]
    p.append(img)
    #Creamos una imagen por nivel deseado
    for i in range(nv):
        #A la imagen actual le aplicamos el filtro gaussiano
        imgauss=applySeparable2DMask(p[i], gaussianMask1D(sigma),gaussianMask1D(sigma),modo_borde=modo_borde)
        #Ahora añadimos la imagen sin las filas y columnas impares
        p.append(downsampling(imgauss, ratiods))
    return p

"""Para hacer cada imagen de la pirámide Laplaciana, seguimos la idea de las
diapositivas: para cada imagen, le hacemos downsampling y aplicamos el filtro
de la gaussiana. Tras ello, hacemos upsampling al tamaño de la imagen anterior
y restamos ésta con la nueva."""
def piramideLaplaciana(img, sigma, nv, modo_borde="reflect", ratious=2):
    #Vector en el que guardamos las imágenes
    plap=[]
    #Hacemos la pirámide gaussiana sin la original y por ello necesitaremos
    #un nivel más para obtener los niveles deseados
    pgauss=piramideGaussiana(img, sigma, nv+1,modo_borde=modo_borde, ratiods=ratious)
    #Por cada nivel pedido hacemos upsampling y restamos con la anterior
    #para obtener la imagen de la piramide
    for i in range(nv+1):
        #La imagen i de la piramide gaussiana
        imgauss=pgauss[i]
        #Imagen i+1 para hacer upsampling y restar luego
        imgsig=upsampling(pgauss[i + 1], ratious)
        #Puede que alguna no sea divisible por el sampling así que habrá que hacer
        #un resize con opencv
        if imgauss.shape!=imgsig.shape:
            imgsig=cv2.resize(imgsig, (imgauss.shape[1], imgauss.shape[0]))
        #Finalmente añadimos la resta de ambas imagenes
        plap.append(imgauss - imgsig)
    #Devolvemos también la última imagen de la piramide gaussiana para reconstruir
    return plap, pgauss[nv]

"""Esta función crea la pirámide de imágenes con el formato visto en clase,
la piramide se contruye con la imagen original a la izquierda y todas las más
pequeñas a su derecha ordenadas de mayor a menor de arriba a abajo"""
def creaImagenPiramide(p):
    #Guardamos las filas y columnas extra para el marco
    filas, colum=p[0].shape[:2]
    diffilas=np.sum([img.shape[0] for img in p[1:]]) - filas
    filas_extra=diffilas if diffilas > 0 else 0
    #La diferencia de columas será lo que ocupe la imagen mas grande sin
    #contar la original
    difcolum=p[1].shape[1]
    colum_extra = difcolum if difcolum > 0 else 0
    #Creamos el marco según si la imagen está en escala de grises o no
    if len(p[0].shape)==2:
        piramide=np.zeros((filas + filas_extra, colum + colum_extra), dtype = np.double)
    else:
        piramide=np.zeros((filas + filas_extra, colum + colum_extra,3), dtype = np.double)
    #Introducimos la primera imagen
    piramide[:filas, :colum]=p[0]
    #Vamos añadiendo el resto de imagenes una debajo de otra al lado de la original
    i = 0
    for im in p[1:]:
        #Extraemos las filas y columas de la imagen que toca
        filasim, columim=im.shape[:2]
        #Pegada a la derecha de la original, la añadimos debajo de la anterior
        piramide[i:i + filasim, colum:colum + columim]=im
        i+=filasim

    return piramide

"""A partir de la última imagen de la pirámide gaussiana y haciendo uso de la
laplaciana se reconstruye la imagen original como se explicó en teoría: 
redimensionando y sumando con cada imagen de la piramide laplaciana"""
def reconstruirImgLaplPir(img, pir):
    #Recorremos de forma inversa la pirámide
    for i in range (len(pir)-1,-1,-1):
        #Redimensionamos a la imagen i-ésima de la laplaciana
        img = cv2.resize(img, (pir[i].shape[1], pir[i].shape[0]))
        img = img + pir[i]
        
    return img

"""Hibrida dos imágenes una con filtro paso bajo (gaussiano) y otra
con filtro paso alto (1-gaussiano)"""
def hibridar(im1,im2,sigma1,sigma2,modo_borde="cero"):
    #Imagen con filtro de paso alto
    im_paso_alto = im2 - applySeparable2DMask(im2, gaussianMask1D(sigma2), gaussianMask1D(sigma2), modo_borde=modo_borde)
    #Imagen con filtro de paso bajo
    im_paso_bajo = applySeparable2DMask(im1, gaussianMask1D(sigma2), gaussianMask1D(sigma2), modo_borde=modo_borde)
    #La imagen híbrida es la suma de ambas
    imhib = im_paso_bajo + im_paso_alto
    #Mostramos las imágenes alta, baja e híbrida en una misma ventana con las
    #imágenes normalizadas en el [0,1]
    res = cv2.hconcat([normMatrix(im_paso_bajo), normMatrix(im_paso_alto), normMatrix(imhib)])
    return imhib, res

"""Esta función hibridará por completo todas las parejas de imágenes de la
carpeta data a través de la función hibridar"""
def ejercicio3():
    sigmaMuestra=1.5
    #Gato y perro
    sigmab = 8.0
    sigmaa = 5.0
    #Leemos en blanco y negro las imágenes
    im1 = cv2.imread("./imagenes/cat.bmp", 0)
    im2 = cv2.imread("./imagenes/dog.bmp", 0)
    #Hibridamos y pintamos las tres imagenes juntas
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "Gato/Perro hibrido")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Gato/Perro")
    
    #Moto y bicicleta
    sigmab = 8.0
    sigmaa = 3.0
    #Leemos en blanco y negro las imágenes
    im2 = cv2.imread("./imagenes/motorcycle.bmp", 0)
    im1 = cv2.imread("./imagenes/bicycle.bmp", 0)
    #Hibridamos y pintamos las tres imagenes juntas
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "Moto/Bici hibrido")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Moto/bici")
    
    #Pez y submarino
    sigmab = 5.0
    sigmaa = 2.0
    #Leemos en blanco y negro las imágenes
    im2 = cv2.imread("./imagenes/fish.bmp", 0)
    im1 = cv2.imread("./imagenes/submarine.bmp", 0)
    #Hibridamos y pintamos las tres imagenes juntas
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "Pez/submarino hibrida")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Pez/submarino")
    
    #Pájaro y avión
    sigmab = 6.0
    sigmaa = 2.0
    #Leemos en blanco y negro las imágenes
    im2 = cv2.imread("./imagenes/bird.bmp", 0)
    im1 = cv2.imread("./imagenes/plane.bmp", 0)
    #Hibridamos (la función ya se encarga de mostrar las 3 imágenes pedidas)
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "pajaro/avion hibrida")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide pajaro/avion")
    
    #Einstein y Marilyn
    sigmab = 5.0
    sigmaa = 2.0
    #Leemos en blanco y negro las imágenes
    im1 = cv2.imread("./imagenes/marilyn.bmp", 0)
    im2 = cv2.imread("./imagenes/einstein.bmp", 0)
    #Hibridamos (la función ya se encarga de mostrar las 3 imágenes pedidas)
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3,"Einstein/Marilyn hibrida")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Einstein/Marilyn")

    return 0

"""Como el ejercicio 3 pero hibridando a color"""
def bonus1():
    sigmaMuestra=1.5
    #Gato y perro
    sigmab = 8.0
    sigmaa = 5.0
    #Leemos en color las imágenes
    im1 = cv2.imread("./imagenes/cat.bmp", 1)
    im2 = cv2.imread("./imagenes/dog.bmp", 1)
    #Hibridamos y pintamos las tres imagenes juntas
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "Gato/Perro hibrido a color")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Gato/Perro a color")
    
    #Moto y bicicleta
    sigmab = 8.0
    sigmaa = 3.0
    #Leemos en color las imágenes
    im2 = cv2.imread("./imagenes/motorcycle.bmp", 1)
    im1 = cv2.imread("./imagenes/bicycle.bmp", 1)
    #Hibridamos y pintamos las tres imagenes juntas
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "Moto/Bici hibrido a color")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Moto/bici a color")
    
    #Pez y submarino
    sigmab = 5.0
    sigmaa = 2.0
    #Leemos en color las imágenes
    im2 = cv2.imread("./imagenes/fish.bmp", 1)
    im1 = cv2.imread("./imagenes/submarine.bmp", 1)
    #Hibridamos y pintamos las tres imagenes juntas
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "Pez/submarino hibrida a color")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Pez/submarino a color")
    
    #Pájaro y avión
    sigmab = 6.0
    sigmaa = 2.0
    #Leemos en color las imágenes
    im2 = cv2.imread("./imagenes/bird.bmp", 1)
    im1 = cv2.imread("./imagenes/plane.bmp", 1)
    #Hibridamos (la función ya se encarga de mostrar las 3 imágenes pedidas)
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3, "pajaro/avion hibrida a color" )
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide pajaro/avion a color")
    
    #Marilyn y Einstein
    sigmab = 5.0
    sigmaa = 2.0
    #Leemos en color las imágenes
    im1 = cv2.imread("./imagenes/marilyn.bmp", 1)
    im2 = cv2.imread("./imagenes/einstein.bmp", 1)
    #Hibridamos (la función ya se encarga de mostrar las 3 imágenes pedidas)
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3,"Einstein/Marilyn hibrida a color")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, sigmaMuestra, 4))
    pintaIm(res, "Piramide Einstein/Marilyn a color")

    return 0

"""Igual que en el bonus1 pero con dos imágenes que yo he buscado"""
def bonus2():
    sigmab = 4.3
    sigmaa = 2.4
    #Leemos en color las imágenes
    im1 = cv2.imread("./imagenes/tigre.bmp", 1)
    im2 = cv2.imread("./imagenes/lobo.bmp", 1)
    #Reescalamos para que midan igual
    im2 = cv2.resize(im2, (im1.shape[1],im1.shape[0]))
    #Hibridamos (la función ya se encarga de mostrar las 3 imágenes pedidas)
    h, im3 = hibridar(im1, im2, sigmab, sigmaa)
    pintaIm(im3,"Lobo/Tigre hibrida")
    #Creamos la pirámide y dibujamos
    res = creaImagenPiramide(piramideGaussiana(h, 1.5, 4))
    pintaIm(res, "Piramide Lobo/Tigre")
    return 0

print("EJERCICIO 1 ejecución A")
#probar máscara gausiana
print("Frecuencias tomadas de forma aleatoria")
vector1D = np.random.uniform(low=0, high=1, size=(50) )
plt.figure(1)
plt.plot(vector1D)
plt.show()

print("Máscara gaussiana con sigma=2.0")
sigma=2.0
plt.figure(2)
plt.plot(gaussianMask1D(sigma), label="Gaussiana")
plt.legend()
plt.show()  


print("Frecuencias al aplicar la máscara gaussiana")
vector1D_gaussiana = apply1DMask(vector1D, gaussianMask1D(sigma),sigma)   
plt.figure(3) 
plt.plot(vector1D_gaussiana, label="Gaussian mask")
plt.legend()
plt.show()

print("Máscara gaussiana derivada con sigma = 2.0")
sigma=2.0
plt.figure(2)
plt.plot(derivGaussianMask1D(1,sigma), label="Derivada gaussiana")
plt.legend()
plt.show()  

print("Frecuencias al aplicar la máscara gaussiana derivada")
vector1D_gaussiana = apply1DMask(vector1D, derivGaussianMask1D(1,sigma),sigma)   
plt.figure(3) 
plt.plot(vector1D_gaussiana, label="Deriv gaussian mask")
plt.legend()
plt.show()

print("Máscara gaussiana segunda derivada con sigma = 2.0")
sigma=2.0
plt.figure(2)
plt.plot(derivGaussianMask1D(2,sigma), label="Segunda derivada gaussiana")
plt.legend()
plt.show()  

print("Frecuencias al aplicar la máscara gaussiana doble derivada")
vector1D_gaussiana = apply1DMask(vector1D, derivGaussianMask1D(2,sigma),sigma)   
plt.figure(3) 
plt.plot(vector1D_gaussiana, label="Second deriv gaussian mask")
plt.legend()
plt.show()


input("\n--- Pulsar tecla para continuar. Ejecucion B ---\n")

sigma=2.0
print("Imagen gato original")
im=cv2.imread("./imagenes/cat.bmp",-1)
im=pintaIm(im,"original cat")

print("Gato con el filtro de gaussiana 2D")
im=cv2.imread("./imagenes/cat.bmp",-1)
im=applySeparable2DMask(im, gaussianMask1D(sigma), gaussianMask1D(sigma))
im=pintaIm(im,"cat gaussian")

print("Gato con el filtro de gaussiana 2D con sigma grande y bordes reflejados")
im=cv2.imread("./imagenes/cat.bmp",-1)
im=applySeparable2DMask(im, gaussianMask1D(4), gaussianMask1D(4), modo_borde="reflect")
im=pintaIm(im,"cat gaussian big sigma")

print("Gato en blanco y negro con el filtro de gaussiana 2D")
im=cv2.imread("./imagenes/cat.bmp",0)
im=applySeparable2DMask(im, gaussianMask1D(sigma), gaussianMask1D(sigma))
im=pintaIm(im,"cat gaussian b&w")

print("Gato con el filtro de GaussianBlur")
im=cv2.imread("./imagenes/cat.bmp",0)
#Comparamos con la función de OpenCV GaussianBlur
dst=cv2.GaussianBlur(im,(len(gaussianMask1D(sigma)),len(gaussianMask1D(sigma))),sigma,sigmaY=sigma)
pintaIm(np.hstack((im, dst)),"Gaussian Blur Cat")

print("Gato con el filtro gaussiano derivado")
im=cv2.imread("./imagenes/cat.bmp",-1)
#Aplicamos las máscaras derivadas a la imagen
im=cv2.imread("./imagenes/cat.bmp",-1)
im=applySeparable2DMask(im, derivGaussianMask1D(1,sigma), derivGaussianMask1D(1,sigma))
im=pintaIm(im,"cat gaussian deriv")

print("Gato con el filtro gaussiano derivado dos veces")
im=cv2.imread("./imagenes/cat.bmp",-1)
im=applySeparable2DMask(im, derivGaussianMask1D(2,sigma), derivGaussianMask1D(2,sigma))
im=pintaIm(im,"cat gaussian doble deriv")

input("\n--- Pulsar tecla para continuar. Ejecucion C ---\n")

#Gaussiana sin derivar
sigma=2.0
d0=gaussianMask1D(sigma)
im=cv2.imread("./imagenes/cat.bmp",0)
d1,d2,im=aplicaDerivKernel(im,0,1,int(2*3*sigma+1))
plt.figure(4)
#Dibujamos la máscara gaussiana que nos devuelve la función de OpenCV
plt.plot(d1, label="Deriv gaussiana getDerivKernel sigma=2")
#Dibujamos junto a ella nuestra máscara para comparar
plt.plot(d0, label="Gaussiana sigma=2")
plt.legend()

#Aplicamos a una foto para comparar con nuestras máscaras
pintaIm(im,"Gaussian deriv sigma=2 getDerivKernel")

#Gaussiana sin derivar
sigma=4.0
d02=gaussianMask1D(sigma)
im=cv2.imread("./imagenes/cat.bmp",0)
d12,d22,im=aplicaDerivKernel(im,0,1,int(2*3*sigma+1))
plt.figure(4)
#Dibujamos la máscara gaussiana que nos devuelve la función de OpenCV
plt.plot(d12, label="Deriv gaussiana getDerivKernel sigma=4")
#Dibujamos junto a ella nuestra máscara para comparar
plt.plot(d02, label="Gaussiana sigma=4")
plt.legend()
plt.show()  

#Aplicamos a una foto para comparar con nuestras máscaras
pintaIm(im,"Gaussian deriv sigma=4 getDerivKernel")

#Gaussiana primera derivada
sigma=2.0
d0=derivGaussianMask1D(1,sigma)
im=cv2.imread("./imagenes/cat.bmp",0)
d1,d2,im=aplicaDerivKernel(im,1,1,int(2*3*sigma+1))
plt.figure(5)
#Dibujamos la máscara gaussiana que nos devuelve la función de OpenCV
plt.plot(d1, label="Deriv gaussiana getDerivKernel")
#Dibujamos junto a ella nuestra máscara para comparar
plt.plot(d0, label="Derivada gaussiana")
plt.legend()
plt.show()  

#Aplicamos a una foto para comparar con nuestras máscaras
pintaIm(im,"Gaussian deriv getDerivKernel")

#Gaussiana segunda derivada
sigma=2.0
d0=derivGaussianMask1D(2,sigma)
im=cv2.imread("./imagenes/cat.bmp",0)
d1,d2,im=aplicaDerivKernel(im,2,2,int(2*3*sigma+1))
plt.figure(6)
#Dibujamos la máscara gaussiana que nos devuelve la función de OpenCV
plt.plot(d1, label="Doble deriv gaussiana getDerivKernel")
#Dibujamos junto a ella nuestra máscara para comparar
plt.plot(d0, label="Segunda Derivada gaussiana")
plt.legend()
plt.show() 

#Aplicamos a una foto para comparar con nuestras máscaras
pintaIm(im,"Gaussian doble deriv getDerivKernel")

input("\n--- Pulsar tecla para continuar. Ejecucion D ---\n")


#Probamos la laplaciana con sigma 1 y 3 y bordes de 0 y reflejo

sigma=1.0
im=cv2.imread("./imagenes/cat.bmp",0)
im, lapl=laplaciana(im,sigma,modo_borde="cero")
#Aplicamos a una imagen la laplaciana
pintaIm(im,"Laplaciana sigma=1 borde cero")
#Pintamos también la máscara laplaciana
pintaIm(lapl, "Mascara Laplaciana sigma=1")

sigma=3.0
im=cv2.imread("./imagenes/cat.bmp",0)
im, lapl=laplaciana(im,sigma,modo_borde="cero")
#Aplicamos a una imagen la laplaciana
pintaIm(im,"Laplaciana sigma=3 borde cero")
#Pintamos también la máscara laplaciana
pintaIm(lapl, "Mascara Laplaciana sigma=3")

sigma=1.0
im=cv2.imread("./imagenes/cat.bmp",0)
im, lapl=laplaciana(im,sigma,modo_borde="reflect")
#Aplicamos a una imagen la laplaciana
pintaIm(im,"Laplaciana sigma=1 borde reflejo")


sigma=3.0
im=cv2.imread("./imagenes/cat.bmp",0)
im, lapl=laplaciana(im,sigma,modo_borde="reflect")
#Aplicamos a una imagen la laplaciana
pintaIm(im,"Laplaciana sigma=3 borde reflejo")


print("EJERCICIO 2. Ejecucion A")
input("\n--- Pulsar tecla para continuar ---\n")


im=cv2.imread("./imagenes/cat.bmp",0)
piramide=piramideGaussiana(im, 2.0,4, modo_borde="edge")
img=creaImagenPiramide(piramide)
pintaIm(img, "piramide gaussiana cat b&w")


input("\n--- Pulsar tecla para continuar. Ejecucion B ---\n")

im=cv2.imread("./imagenes/cat.bmp",0)
piramideL, gaussim=piramideLaplaciana(im, 2.0,4,modo_borde="edge")
img=creaImagenPiramide(piramideL)
pintaIm(img, "piramide laplaciana cat b&w")

print("Reconstruccion de la imagen a partir de la piramide")
rebuild=reconstruirImgLaplPir(gaussim, piramideL)
pintaIm(rebuild, "Imagen reconstruida")

input("\n--- Pulsar tecla para continuar ---\n")

print("EJERCICIO 3")
ejercicio3()

input("\n--- Pulsar tecla para continuar ---\n")

print("BONUS 1")
bonus1()

input("\n--- Pulsar tecla para continuar ---\n")

print("BONUS 2")
bonus2()
