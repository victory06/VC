#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:26:52 2020

@author: victor
"""

from matplotlib import pyplot as plt
from math import log
import cv2 
import numpy as np
import random

BLOCK_SIZE = 3       #Tamaño de ventana para esquinas
BLOCK_SIZE_F = 7     #Tamaño de ventana para supresión de no máximos
UMBRAL = 10          #Distancia de 10 entre máximos
np.random.seed(1)

###############################################################################
# Funciones auxiliares #
###############################################################################

#Función propia para normalizar matrices
def normMatrix(im):
    if(im.min()<0 ):
        im=im-im.min()
    if(im.max()!=0):
        im = im / im.max()
    return im


#Función para mostrar imágenes con OpenCV
def pintaIm(im,name):
    im=normMatrix(im)
    cv2.imshow(name,im)
    cv2.waitKey(20)
    return im

"""Normaliza una imagen haciendo uso de OpenCV"""
def normalize(im):

    return cv2.normalize(im, None, 0.0, 1.0, cv2.NORM_MINMAX)

"""Muestra una imagen normalizada usando matplotlib"""
def printIm(im, title = "", show = True, tam = (7,7)):

    show_title = len(title) > 0

    if show:
        fig = plt.figure(figsize = tam)

    im = normalize(im)  # Normalizamos a [0,1]
    plt.imshow(im, interpolation = None, cmap = 'gray')
    plt.xticks([]), plt.yticks([])

    if show:
        if show_title:
            plt.title(title)
        plt.show(block = False)  # No cerrar el plot

"""Muestra un vector de imagenes en una misma imagen."""
def pintaVectorIm(vim, titles = "", ncols = 3, tam = (7,7)):

    show_title = len(titles) > 0

    nfilas = len(vim) // ncols + (0 if len(vim) % ncols == 0 else 1)
    plt.figure(figsize = tam)

    for i in range(len(vim)):
        plt.subplot(nfilas, ncols, i + 1)
        if show_title:
            plt.title(titles[i])
        printIm(vim[i], title="",show=False)

    plt.show(block = False)

        


"""Funcion para leer imagenes, col_flag indica si se leen en color (1) o 
en escala de grises (0)."""
def leeIm(path, col_flag = 0):
    im = cv2.imread(path, col_flag)
    #Si esta en color tenemos que cmabiar a RGB
    if col_flag!=0:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im.astype(np.float32)

"""Devuelve una lista que representa las imagenes que componen una pirámide 
gaussiana.
    - tam: Representa el tamaño de la imagen
    - im: imagen original que no se modifica en la función"""
def gaussianPyramid(im, tam, borde = cv2.BORDER_REPLICATE):
    pyramid = [im]
    for k in range(tam):
        pyramid.append(cv2.pyrDown(pyramid[-1], borderType = borde))

    return pyramid

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

"""Pasa una imagen de gris a RGB con OpenCV."""
def grayToRgb(im):
    im_rgb = cv2.normalize(im.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)

#Funciones para el Ejercicio 1

"""Realiza supresión de no máximos en una imagen. Develve una lista con los índices
       de los puntos que son máximos locales."""
def nonMaxiumSupression(f, W_SIZE):

    nfilas, ncols = f.shape[:2]
    indice_max = []
    d = W_SIZE // 2

    for x in range(nfilas):
        for y in range(ncols):
            #Saltamos el punto su no supera el umbral (distancia entre máximos)
            if f[x, y] <= UMBRAL:
                continue

            #Seleccionamos los vecinos en un cuadrado de lado 'BLOCK_SIZE'
            t = x - d if x - d >= 0 else 0
            b = x + d + 1
            l = y - d if y - d >= 0 else 0
            r = y + d + 1
            window = f[t:b, l:r]
            max_window = np.amax(window)

            #Vemos si es máximo local
            if max_window <= f[x, y]:
                indice_max.append((x, y))
                window[:] = 0
                f[x, y] = max_window

    return indice_max


"""Realiza la detección de puntosHarris. Devuelve las imágenes con los puntos dibujados 
en cada escala, la imagen original con todos los puntos, y un vector de los puntos detectados."""
def harrisDetection(im, levels, nombre):
    im_scale_kp = []
    keypoints_orig = []
    window = BLOCK_SIZE_F
    num_points = [1500, 400, 100]        #Obtengo 2000 puntos con proporciones 70/25/5%
    escalas = [3,7,15]

    # Pirámide gaussiana de la imagen
    orig_filas, orig_cols = im.shape[:2]
    pyramid = gaussianPyramid(im, levels)
    
    print("Pirámide Gaussiana de "+nombre)
    pintaIm(creaImagenPiramide(pyramid), "Piramide Gaussiana de "+nombre)
    
    input("\n--- Pulsar tecla para continuar ---\n")
  
    #Para cada escala detectamos los puntos
    for s in range(len(pyramid)):
        im_scale = pyramid[s]
        nfilas, ncols = im_scale.shape[:2]

        #Extraemos la información de cada píxel
        dst = cv2.cornerEigenValsAndVecs(im_scale, BLOCK_SIZE, ksize = 3)

        #Calculamos la matriz con los valores para el criterio de Harris
        f = np.empty_like(im_scale)
        for x in range(nfilas):
            for y in range(ncols):
                l1 = dst[x, y, 0]
                l2 = dst[x, y, 1]
                f[x, y] = (l1 * l2) / (l1 + l2) if l1 + l2 != 0.0 else 0.0
                
        
        print("Valores para el criterio de Harris en"+nombre+" en el nivel "+str(s))    
        pintaIm(f,"Matriz de valores para el criterio de Harris en"+nombre+" en el nivel "+str(s))  
        input("\n--- Pulsar tecla para continuar ---\n")
        #Apartado b
        #Supresion de no maximos y ordenamos por intensidad el vector
        indice_max = nonMaxiumSupression(f, window)
        indice_max = sorted(indice_max, key = lambda x: f[x], reverse = True)
        
        #Alisamos la imagen
        img_G = cv2.GaussianBlur(im, ksize = (0, 0), sigmaX = 4.5)
    
        img_dx = gaussianPyramid(cv2.Sobel(img_G, -1, 1, 0), 2)
        img_dy = gaussianPyramid(cv2.Sobel(img_G, -1, 0, 1), 2)
        
        #Nos quedamos con los 'num_points' puntos de mayor intensidad
        list_keypoints = []
        n_points = num_points[s]
        
        for p in indice_max[:n_points]:
                #Calculamos la orientación de los puntos
                norm = np.sqrt(img_dx[0][p] * img_dx[0][p] + img_dy[0][p] * img_dy[0][p])
                angle_sin = img_dy[0][p] / norm if norm > 0 else 0.0
                angle_cos = img_dx[0][p] / norm if norm > 0 else 0.0
                angle = np.degrees(np.arctan2(angle_sin, angle_cos)) + 180
    
                #Creamos una estructura KeyPoint con cada punto que queda
                list_keypoints.append(cv2.KeyPoint(p[1], p[0], 1.3*3, _angle = angle))
    
                #También con sus coordenadas respecto a la imagen original
                keypoints_orig.append(cv2.KeyPoint((2**s) * p[1], (2**s) * p[0], 1.3*escalas[s], _angle = angle))
        
        #Pasamos la imagen a color y dibujamos cada KeyPoint
        img_rgb = grayToRgb(im_scale)
        img_kp = cv2.drawKeypoints(img_rgb, list_keypoints, np.array([]),
                              flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        input("\n--- Pulsar tecla para continuar ---\n")
        print("Puntos detectados en la octava " + str(s) + ": " + str(len(list_keypoints)))
        
        im_scale_kp.append(img_kp.astype(np.float32)[:orig_filas // 2**s, :orig_cols // 2**s])
        #Reducimos el tamaño de la ventana supresión de no máximos 
        #al reducir la escala.
        #Los tamaños de las ventanas quedan: 7x7, 5x5 y 3x3.
        if window > 3:
            window = window - 2
    
    #Pintamos los KeyPoints en la imagen original
    img_rgb = grayToRgb(im)
    im_kp = cv2.drawKeypoints(img_rgb, keypoints_orig, np.array([]),
                              flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_kp = im_kp.astype(np.float32)[:orig_filas, :orig_cols]

    print("Puntos totales detectados: " + str(len(keypoints_orig)) + "\n")

    return im_scale_kp, im_kp, keypoints_orig

"""Realiza un doble refinamiento de las esquinas. Devuelve las esquinas refinadas
y un vector con tres imágenes donde se muestra en rojo la anterior detección y en
verde la refinada."""
def refineCorners(im, keypoints):
    ZOOM = 7
    res = []
    win_size = (7, 7)           #Ventana para el primer refinamiento
    win_size2 = (5, 5)          #Ventana para el segundo refinamiento
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    points = np.array([p.pt for p in keypoints], dtype = np.uint32)
    esquinas = points.reshape(len(keypoints), 1, 2).astype(np.float32)

    #Coordenadas subpixel de los KeyPoints con tamaño de ventana 7x7
    cv2.cornerSubPix(im, esquinas, win_size, zero_zone, criteria)

    #Para las dos capas superiores volvemos a hacer un refinamiento con venatana 5x5
    esquinas2 = esquinas[-1400:] 
    cv2.cornerSubPix(im, esquinas2, win_size2, zero_zone, criteria)
    esquinas = np.concatenate((esquinas[1400:], esquinas2), axis=0)

    
    #Elegimos tres puntos aleatoriamente cuyas coordenadas difieran
    selected_points = []
    count = 0
    while count < 3:
        index = random.randint(0, len(points) - 1)
        if index not in selected_points and \
           (points[index][:2] != esquinas[index][0][:2]).any():
            selected_points.append(index)
            count = count + 1

    for index in selected_points:
        #Recuperamos las coordenadas originales e interpoladas
        y, x = points[index][:2]
        ry, rx = esquinas[index][0][:2]

        #Pasamos la imagen original a color para dibujar sobre ella
        im_rgb = cv2.normalize(im.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB).astype(np.float32)

        #Seleccionamos una ventana 9x9 alrededor del punto original
        t = x - 4 if x - 4 >= 0 else 0
        b = x + 4 + 1
        l = y - 4 if y - 4 >= 0 else 0
        r = y + 4 + 1
        window = im_rgb[t:b, l:r]

        #Interpolamos con zoom de 10x
        window = cv2.resize(window, None, fx = ZOOM, fy = ZOOM)

        #En rojo el punto original en el centro
        window = cv2.circle(window, (ZOOM * 4 + 1, ZOOM * 4 + 1), 3, (255, 0, 0))

        #En verde el punto corregido
        window = cv2.circle(window, (int(ZOOM * (4 + ry - y) + 1), int(ZOOM * (4 + rx - x) + 1)), 3, (0, 255, 0))

        res.append(window)

    return esquinas, res

#Funciones para Ejercicio 2

"""Devuelve los keypoints y descriptores AKAZE de una imagen."""
def akazeDescriptor(im):
    return cv2.AKAZE_create().detectAndCompute(im, None)

"""Devuelve los matches entre los descriptores de dos imágenes por el
método de fuerza bruta más crossCheck"""
def matchesBruteforce(desc1, desc2):
    #Creamos el objeto BFMatcher más crossCheck
    bf = cv2.BFMatcher_create(normType = cv2.NORM_HAMMING, crossCheck = True)
    #Calculamos los matches entre descriptores
    matches = bf.match(desc1, desc2)

    return matches

"""Devuelve los matches entre los descriptores de dos imágenes por el
método de Lowe-Average-2NN"""
def matchesLowe2nn(desc1, desc2):
    #Objeto BFMatcher
    bfMatch = cv2.BFMatcher_create(normType = cv2.NORM_HAMMING)
    #Encontramos los 2 mejores matches entre los descriptores de las imágenes
    matches = bfMatch.knnMatch(desc1, desc2, k = 2)
    #Descartamos correspondencias ambiguas siguiendo el criterio de Lowe
    selected = []
    for m1, m2 in matches:
        if m1.distance < 0.75 * m2.distance:
            selected.append(m1)

    return selected

"""Devuelve dos imágenes con las correspondencias entre los keypoints
   extraídos de las dos immágenes que se pasan por parámetro, im1 e im2,
   usando el descriptor AKAZE: con el método BruteForce + crossCheck 
   y con Lowe-Average-2NN."""
def getMatches(im1, im2):
    #Descriptores y keypoints de AKAZE
    kp1, desc1 = akazeDescriptor(im1)
    kp2, desc2 = akazeDescriptor(im2)
    #100 matches aleatorios con BruteForce + crossCheck
    matches_bf = matchesBruteforce(desc1, desc2)
    n = min(100, len(matches_bf))
    matches_bf = random.sample(matches_bf, n)
    #Pintamos las corresondencias
    im_matches_bf = cv2.drawMatches(im1, kp1, im2, kp2, matches_bf, None,
                                    flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    #Obtenemos 100 matches aleatorios con Lowe-Average-2NN
    match_2nn = matchesLowe2nn(desc1, desc2)
    n = min(100, len(match_2nn))
    match_2nn = random.sample(match_2nn, n)
    #Pintamos las corresondencias
    im_matches_2nn = cv2.drawMatches(im1, kp1, im2, kp2, match_2nn, None,
                                    flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    return im_matches_bf.astype(np.float32), im_matches_2nn.astype(np.float32)


#Funciones para el Ejercicio 3

"""Calcula una homografía de im1 a im2 haciendo uso de findHomography
   de OpenCV."""
def getHomography(im1, im2):
    kp1, desc1 = akazeDescriptor(im1)
    kp2, desc2 = akazeDescriptor(im2)
    matches = matchesLowe2nn(desc1, desc2)
    query = np.array([kp1[match.queryIdx].pt for match in matches])
    train = np.array([kp2[match.trainIdx].pt for match in matches])

    return cv2.findHomography(query, train, cv2.RANSAC)[0]


#Funciones para el Ejercicio 4

def ejercicio4(ims, h, w):

    central = len(ims) // 2

    # Definimos un canvas
    canvas = np.zeros((h, w, 3), dtype = np.float32)

    # Calculamos la homografía (traslación) que lleva la imagen central al mosaico
    tx = (w - ims[central].shape[1]) / 2
    ty = (h - ims[central].shape[0]) / 2
    H0 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Trasladamos la imagen central al mosaico
    canvas = cv2.warpPerspective(ims[central], H0, (w, h), dst = canvas, borderMode = cv2.BORDER_TRANSPARENT)

    # Calculamos las homografías entre cada dos imágenes
    homographies = []
    for i in range(len(ims)):
        if i != central:
            j = i + 1 if i < central else i - 1
            homographies.append(getHomography(ims[i], ims[j]))

        else: # No se usa la posición central
            homographies.append(np.array([]))

    # Trasladamos el resto de imágenes al mosaico
    H = H0
    G = H0
    for i in range(central)[::-1]:
        H = H @ homographies[i]
        canvas = cv2.warpPerspective(ims[i], H, (w, h), dst = canvas, borderMode = cv2.BORDER_TRANSPARENT)

        j = 2 * central - i
        if j < len(ims):
            G = G @ homographies[j]
            canvas = cv2.warpPerspective(ims[j], G, (w, h), dst = canvas, borderMode = cv2.BORDER_TRANSPARENT)

    # Mostramos el mosaico
    printIm(canvas)

"""Ejecucion secuencial de los apartados."""

###############################################################################
# EJERCICIO 1 #
###############################################################################


np.random.seed(1)

# Leer imágenes de Yosemite
img1 = leeIm('imagenes/Yosemite1.jpg', 0)
img2 = leeIm('imagenes/Yosemite2.jpg', 0)

print("Detectando puntos Harris en yosemite1.jpg...\n")
h1_scale, h1_orig, h1_keypoints = harrisDetection(img1, 2, "Yosemite1")
pintaIm(h1_orig, "Yosemite1 Harris detection")
pintaIm(creaImagenPiramide(h1_scale),"Harris detection piramide Yosemite1")

print("\nMostrando coordenadas subpíxel corregidas en yosemite1.jpg...\n")
refinados, subpix1 = refineCorners(img1, h1_keypoints)
pintaVectorIm(subpix1)

input("\n--- Pulsar tecla para continuar. Yosemite2 ---\n")
print("Detectando puntos Harris en yosemite2.jpg...\n")
h2_scale, h2_orig, h2_keypoints = harrisDetection(img2, 2, "Yosemite2")
pintaIm(h2_orig, "Yosemite2 Harris detection")
pintaIm(creaImagenPiramide(h2_scale),"Harris detection piramide Yosemite2")

print("\nMostrando coordenadas subpíxel corregidas en yosemite1.jpg...\n")
refinados, subpix2 = refineCorners(img2, h2_keypoints)
pintaVectorIm(subpix2)


input("\n--- Pulsar tecla para continuar. Ejercicio 2 ---\n")
###############################################################################
# EJERCICIO 2 #
###############################################################################


# Leer imágenes de Yosemite
img1 = leeIm('imagenes/Yosemite1.jpg', 0).astype(np.uint8)
img2 = leeIm('imagenes/Yosemite2.jpg', 0).astype(np.uint8)

im_matches_bf, im_matches_2nn = getMatches(img1, img2)

print("Correspondencias con BruteForce + crossCheck en yosemite...\n")
printIm(im_matches_bf)

print("\nCorrespondencias con Lowe-Average-2NN en yosemite...\n")
printIm(im_matches_2nn)

img1 = leeIm('imagenes/mosaico002.jpg', 0).astype(np.uint8)
img2 = leeIm('imagenes/mosaico003.jpg', 0).astype(np.uint8)

im_matches_bf, im_matches_2nn = getMatches(img1, img2)

print("Correspondencias con BruteForce + crossCheck en mosaico...\n")
printIm(im_matches_bf)

print("\nCorrespondencias con Lowe-Average-2NN en mosaico...\n")
printIm(im_matches_2nn)


input("\n--- Pulsar tecla para continuar. Ejercicio 3 ---\n")

###############################################################################
# EJERCICIO 3 #
###############################################################################

im1 = leeIm('imagenes/Yosemite1.jpg', 1).astype(np.float32)
im2 = leeIm('imagenes/Yosemite2.jpg', 1).astype(np.float32)

# Definimos un canvas suponiendo que las imágenes tienen la misma altura
h, w = im1.shape[0], 940
canvas = np.zeros((h, w, 3), dtype = np.float32)

# La homografía que lleva la primera imagen al mosaico es la identidad
canvas[:im1.shape[0], :im1.shape[1]] = im1

# Calculamos la homografía de la segunda imagen a la primera
H21 = getHomography(im2, im1)

# Trasladamos la segunda imagen al mosaico
canvas = cv2.warpPerspective(im2, H21, (w, h), dst = canvas, borderMode = cv2.BORDER_TRANSPARENT)

print("\nMosaico de 2 imagenes\n")
# Mostramos el mosaico
printIm(canvas)
input("\n--- Pulsar tecla para continuar. ---\n")
im1 = leeIm('imagenes/mosaico002.jpg', 1).astype(np.float32)
im2 = leeIm('imagenes/mosaico003.jpg', 1).astype(np.float32)
im3 = leeIm('imagenes/mosaico004.jpg', 1).astype(np.float32)

# Definimos un canvas suponiendo que las imágenes tienen la misma altura
h, w = im1.shape[0], 500
canvas = np.zeros((h, w, 3), dtype = np.float32)

# La homografía que lleva la primera imagen al mosaico es la identidad
canvas[:im1.shape[0], :im1.shape[1]] = im1

# Calculamos la homografía de la segunda imagen a la primera
H21 = getHomography(im2, im1)

# Trasladamos la segunda imagen al mosaico
canvas = cv2.warpPerspective(im2, H21, (w, h), dst = canvas, borderMode = cv2.BORDER_TRANSPARENT)

h, w = canvas.shape[0], 500
canvas2 = np.zeros((h, w, 3), dtype = np.float32)

canvas2[:canvas.shape[0], :canvas.shape[1]] = canvas

H21 = getHomography(im3, canvas)

canvas2 = cv2.warpPerspective(im3, H21, (w, h), dst = canvas2, borderMode = cv2.BORDER_TRANSPARENT)
print("\nMosaico de 3 imagenes\n")
# Mostramos el mosaico
printIm(canvas2)

input("\n--- Pulsar tecla para continuar. Ejercicio 4 ---\n")
###############################################################################
# EJERCICIO 4 #
###############################################################################


print("\nMosaico002-011...\n")
ims = [leeIm("imagenes/" + "{}00{}.jpg".format("mosaico", i), 1) for i in range(2, 10)] + \
      [leeIm("imagenes/" + "{}0{}.jpg".format("mosaico", i), 1) for i in range(10, 12)]
ejercicio4(ims, 550, 900)

