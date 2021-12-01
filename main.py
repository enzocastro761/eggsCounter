
"""import cv2 #importo opencv
import numpy as np
import imutils #importo imutils

cap = cv2.VideoCapture('cintaTransportadora2.mp4') #leo el video de entrada

fgbg =cv2.createBackgroundSubtractorMOG2()  # Sustraccion de fondo - Porque es python 3, sino seria fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #Para mejorar la imagen binaria obtenida luego de aplicar la sustraccion de fondo
eggs_counter = 0 #contador de huevos

while True:

    ret, frame = cap.read()
    original = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    canny = cv2.Canny(frame, 50, 150)
    if ret == False: break
    frame = imutils.resize(frame, width=640) #cantidad de pixeles (se puede redimensionar)

    # Especificamos los puntos extremos del área a analizar
    area_pts = np.array([[55, 230], [frame.shape[1]-100, 230], [frame.shape[1]-100, 290], [55, 290]]) #Para especificar la zona de interes, es decir donde vamos a analizar cuantos autos pasan

    # Con ayuda de una imagen auxiliar, determinamos el área
    # sobre la cual actuará el detector de movimiento
    imAux = np.zeros(shape=(canny.shape[:2]), dtype=np.uint8) #imagen auxiliar de ceros, del mismo ancho y alto que frame
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1) #se dibuja el area de interes en color blanco (no se ve en el frame)
    image_area = cv2.bitwise_and(canny, canny, mask=imAux) #para que se vean los autos en el area blanca

    # Obtendremos la imagen binaria donde la región en blanco representa
    # la existencia de movimiento
    fgmask = fgbg.apply(image_area) #se ve en blanco los autos en movimiento en el area de interes
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) #transformaciones morfologicas para mejorar la imagen binariav (descarta regiones muy pequeñas de blanco)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    #fgmask = cv2.dilate(fgmask, None, iterations=4) #5 iteraciones para la dilatacion, es para conectar las areas blancas mas grandes que representan al auto
    #fgmask = cv2.erode(fgmask, kernel, iterations=2)
    # Encontramos los contornos presentes de fgmask, para luego basándonos
    # en su área poder determinar si existe movimiento (autos)
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] #para encontrar los contornos presentes en la imagen

    for cnt in cnts: #se analizan los contornos encontrados
        if cv2.contourArea(cnt) > 1000: #se compara con sus areas en pixeles (el valor 1500 es a prueba y error)
            x, y, w, h = cv2.boundingRect(cnt) #encuentra el ancho y alto del contorno
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 1) #dibujo el rectangulo alrededor de los autos

            # Si el auto ha cruzado entre 440 y 460 abierto, se incrementará
            # en 1 el contador de autos
            if 234 < (x + w) < 254: #si el auto pasa por la linea amarilla, sera contado
                eggs_counter = eggs_counter + 1
                cv2.line(frame, (55, 244), (230, 244), (0, 255, 0), 3) #se dibuja una linea verde cuando los autos pasan
#
    # Visualización del conteo de huevos
    #(contornos, _) = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(original, contornos, -1, (0, 0, 255), 2)
    cv2.drawContours(original, [area_pts], -1, (255, 0, 255), 2) #visualizamos el area determinada anteriormente
    cv2.line(original, (55, 260), (original.shape[1]-100, 260), (0, 255, 255), 1) #linea horizontal amarilla ubicada en 244 (eje y)
    cv2.rectangle(original, (original.shape[1]-70, 215), (original.shape[1]-5, 270), (0, 255, 0), 2) #rectangulo donde va el contador, en color verde
    cv2.putText(original, str(eggs_counter), (original.shape[1]-55, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2) #muestro el contador en el rectangulo establecido anteriormente

    #cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask) #muestro la ventanita auxiliar
    cv2.imshow('original', original)

    k = cv2.waitKey(30) & 0xFF #el 70 es para que el video se vea mas lento, si disminuye va mas rapido
    if k ==27: #si apreto la tecla ESC sale.
        break

cap.release()
cv2.destroyAllWindows()

"""

import cv2
import numpy as np
import imutils #importo imutils

sensitivity = 19
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])

cap = cv2.VideoCapture('cintaTransportadora2.mp4') #leo el video de entrada

fgbg =cv2.createBackgroundSubtractorMOG2()  # Sustraccion de fondo - Porque es python 3, sino seria fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #Para mejorar la imagen binaria obtenida luego de aplicar la sustraccion de fondo
eggs_counter = 0 #contador de huevos

while True:

    ret, imagen = cap.read()

    if not ret:    #si no puede capturar imagen entonces sale del while (es para evitar errores)
        break
    imagen = imutils.resize(imagen, width=640)  # cantidad de pixeles (se puede redimensionar)
    imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV) #convierto de BGR a HSV
    area_pts = np.array([[55, 50], [imagenHSV.shape[1] - 100, 50], [imagenHSV.shape[1] - 100, 110], [55, 110]])
	#Detectando colores
    gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    #Eliminamos el ruido
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    #Conertimos la imagen a blanco y negro, a partir de un umbra q se calcula automaticamente
    thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.erode(thresh, kernel, iterations=2)

	#Encontrando contornos
	#OpenCV 4
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # para encontrar los contornos presentes en la imagen

    for cnt in cnts:  # se cuentan los contornos encontrados si estan en el area a analizar
        if cv2.contourArea(cnt) > 7:  # se compara con sus areas en pixeles (el valor 1500 es a prueba y error)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(imagen, center, radius, (255, 0, 0), 2)
            # Si el auto ha cruzado entre 440 y 460 abierto, se incrementará
            # en 1 el contador de autos
            if 78 < int(y)  < 82:  # si el auto pasa por la linea amarilla, sera contado
                eggs_counter = eggs_counter + 1
                if radius > 30:
                    eggs_counter = eggs_counter + 1
                cv2.line(imagen, (55, 80), (imagen.shape[1] - 100, 80), (0, 255, 0),3)  # se dibuja una linea verde cuando los autos pasan

    cv2.drawContours(imagen, [area_pts], -1, (255, 0, 255), 2)  # visualizamos el area determinada anteriormente
    cv2.line(imagen, (55, 50), (55, 110), (0, 255, 255),1)  # linea vertical 1
    #espacio 1: x=95 ; y=80
    cv2.line(imagen, (135, 50), (135, 110), (0, 255, 255),1)  # linea vertical 2
    #espacio 2: x=175 ; y=80
    cv2.line(imagen, (215, 50), (215, 110), (0, 255, 255), 1)  # linea vertical 3
    #espacio 3: x=255 ; y=80
    cv2.line(imagen, (295, 50), (295, 110), (0, 255, 255), 1)  # linea vertical 4
    #espacio 4: x=335 ; y=80
    cv2.line(imagen, (375, 50), (375, 110), (0, 255, 255), 1)  # linea vertical 5
    #espacio 5: x=415 ; y=80
    cv2.line(imagen, (455, 50), (455, 110), (0, 255, 255), 1)  # linea vertical 6
    #espacio 6: x=495 ; y=80
    cv2.line(imagen, (535, 50), (535, 110), (0, 255, 255), 1)  # linea vertical 7
    cv2.line(imagen, (55, 80), (imagen.shape[1] - 100, 80), (0, 255, 255),1)  # linea horizontal amarilla ubicada en 260 (eje y)
    cv2.rectangle(imagen, (imagen.shape[1] - 70, 215), (imagen.shape[1] - 5, 270), (0, 255, 0),2)  # rectangulo donde va el contador, en color verde
    cv2.putText(imagen, str(eggs_counter), (imagen.shape[1] - 55, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('maskBlanco', thresh)
    cv2.imshow('Imagen', imagen)

    k = cv2.waitKey(70) & 0xFF #el 70 es para que el video se vea mas lento, si disminuye va mas rapido
    if k == 27:  # si apreto la tecla ESC sale.
        break

cap.release()
cv2.destroyAllWindows()
"""
import cv2
import numpy as np
import imutils #importo imutils

sensitivity = 21
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])

cap = cv2.VideoCapture('cintaTransportadora2.mp4') #leo el video de entrada

fgbg =cv2.createBackgroundSubtractorMOG2()  # Sustraccion de fondo - Porque es python 3, sino seria fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #Para mejorar la imagen binaria obtenida luego de aplicar la sustraccion de fondo
eggs_counter = 0 #contador de huevos

while True:

    ret, imagen = cap.read()

    if not ret:    #si no puede capturar imagen entonces sale del while (es para evitar errores)
        break
    imagen = imutils.resize(imagen, width=640)  # cantidad de pixeles (se puede redimensionar)
    imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV) #convierto de BGR a HSV
    area_pts = np.array([[55, 50], [imagenHSV.shape[1] - 100, 50], [imagenHSV.shape[1] - 100, 110], [55, 110]])
	#Detectando colores
    maskBlanco = cv2.inRange(imagenHSV, lower_white, upper_white)
    #Las siguientes transformaciones dependen de la iluminacion
    maskBlanco = cv2.morphologyEx(maskBlanco, cv2.MORPH_OPEN, kernel) #transformaciones morfologicas para mejorar la imagen binariav (descarta regiones muy pequeñas de blanco)
    maskBlanco = cv2.morphologyEx(maskBlanco, cv2.MORPH_CLOSE, kernel)
    maskBlanco = cv2.erode(maskBlanco, kernel, iterations=5)
    maskBlanco = cv2.dilate(maskBlanco, None, iterations=3) #5 iteraciones para la dilatacion, es para conectar las areas blancas mas grandes que representan al auto


	#Encontrando contornos
	#OpenCV 4
    cnts = cv2.findContours(maskBlanco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # para encontrar los contornos presentes en la imagen

    for cnt in cnts:  # se cuentan los contornos encontrados si estan en el area a analizar
        x, y, w, h = cv2.boundingRect(cnt)  # encuentra el ancho y alto del contorno
        if (cv2.contourArea(cnt) > 15) and (50 < (y + h) < 130):  # se compara con sus areas en pixeles (el valor 1500 es a prueba y error) y si esta en el area de analisis
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radio = 2
            if 55 < int(y) < 110:   #dibujo la circunferencia azul solo en el rectangulo de interes
                cv2.circle(imagen, center, radio, (255, 0, 0), 3)   #dibujo el punto en cada huevo
                if 78 < int(y)  < 82:  # si el auto pasa por la linea amarilla, sera contado
                    if (55 < int(x) <= 135) and (espacio1 == 0): #si pasa por el espacio 1 por primera vez
                        eggs_counter = eggs_counter + 1
                        espacio1 = 1
                    if (135 < int(x) <= 215) and (espacio2 == 0): #si pasa por el espacio 2 por primera vez
                        eggs_counter = eggs_counter + 1
                        espacio2 = 1
                    if (215 < int(x) <= 295) and (espacio3 == 0):  # si pasa por el espacio 3 por primera vez
                        eggs_counter = eggs_counter + 1
                        espacio3 = 1
                    if (295 < int(x) <= 375) and (espacio4 == 0):  # si pasa por el espacio 4 por primera vez
                        eggs_counter = eggs_counter + 1
                        espacio4 = 1
                    if (375 < int(x) <= 455) and (espacio5 == 0):  # si pasa por el espacio 5 por primera vez
                        eggs_counter = eggs_counter + 1
                        espacio5 = 1
                    if (455 < int(x) <= 535) and (espacio6 == 0):  # si pasa por el espacio 6 por primera vez
                        eggs_counter = eggs_counter + 1
                        espacio6 = 1
                    if radius > 30:
                        eggs_counter = eggs_counter + 1
                    cv2.line(imagen, (55, 80), (imagen.shape[1] - 100, 80), (0, 255, 0),3)  # se dibuja una linea verde cuando los autos pasan
    espacio1 = 0
    espacio2 = 0
    espacio3 = 0
    espacio4 = 0
    espacio5 = 0
    espacio6 = 0
    cv2.drawContours(imagen, [area_pts], -1, (255, 0, 255), 2)  # visualizamos el area determinada anteriormente
    cv2.line(imagen, (55, 50), (55, 110), (0, 255, 255),1)  # linea vertical 1
    #espacio 1: x=95 ; y=80
    cv2.line(imagen, (135, 50), (135, 110), (0, 255, 255),1)  # linea vertical 2
    #espacio 2: x=175 ; y=80
    cv2.line(imagen, (215, 50), (215, 110), (0, 255, 255), 1)  # linea vertical 3
    #espacio 3: x=255 ; y=80
    cv2.line(imagen, (295, 50), (295, 110), (0, 255, 255), 1)  # linea vertical 4
    #espacio 4: x=335 ; y=80
    cv2.line(imagen, (375, 50), (375, 110), (0, 255, 255), 1)  # linea vertical 5
    #espacio 5: x=415 ; y=80
    cv2.line(imagen, (455, 50), (455, 110), (0, 255, 255), 1)  # linea vertical 6
    #espacio 6: x=495 ; y=80
    cv2.line(imagen, (535, 50), (535, 110), (0, 255, 255), 1)  # linea vertical 7
    cv2.line(imagen, (55, 80), (imagen.shape[1] - 100, 80), (0, 255, 255),1)  # linea horizontal amarilla ubicada en 260 (eje y)
    cv2.rectangle(imagen, (imagen.shape[1] - 70, 215), (imagen.shape[1] - 5, 270), (0, 255, 0),2)  # rectangulo donde va el contador, en color verde
    cv2.putText(imagen, str(eggs_counter), (imagen.shape[1] - 55, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('maskBlanco', maskBlanco)
    cv2.imshow('Imagen', imagen)

    k = cv2.waitKey(60) & 0xFF #el 70 es para que el video se vea mas lento, si disminuye va mas rapido
    if k == 27:  # si apreto la tecla ESC sale.
        break

cap.release()
cv2.destroyAllWindows()"""