import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import inspect
import transformaciones as tr

path_img_grayscale = "img_grayscale" # aqui se van a guardar las imagenes en escala de grises
                                     # asi como sus histogramas

print("Creando carpeta de escala de grises ...\n")

if os.path.exists(os.path.join("..", path_img_grayscale)):
    print("\tLa carpeta ya existia")
else:
    os.makedirs(os.path.join("..", path_img_grayscale))
    print("\tSe creo la carpeta exitosamente\n")

print("\nLeyendo imagenes ...\n")

path_img = "img"

imgs = os.listdir(os.path.join("..", path_img))

if len(imgs) == 0:
    print("\tNo se encontraron imagenes, saliendo ...")
    sys.exit(1) # error

print("\texisten imagenes")

print("\nConvirtiendo imagenes a escala de grises ...\n")

for img in imgs:
    if not img.endswith((".jpeg", ".jpg", ".tiff", ".png")): # solo permito estas extensiones
        continue

    print(f"\timagen: {img}")

    img_cv2 = cv2.imread(os.path.join("..", path_img, img)) # la imagen ahora es un arreglo de enteros

    img_cv2 = img_cv2.astype(float) # ahora son flotantes

    promedio = (img_cv2[:, :, 0] + img_cv2[:, :, 1] + img_cv2[:, :, 2]) / 3 # promedia los canales

    img_gris = promedio.astype(np.uint8) # enteros de nuevo, pero escala de grises

    ruta_grayscale = os.path.join("..", path_img_grayscale, img) # ruta guarda en la nueva carpeta
    
    cv2.imwrite(ruta_grayscale, img_gris)

    print("\tescala de grises guardado!")

    # aplanar la matriz 2D a lista 1D
    datos_pixel = img_gris.ravel()

    # crear un nuevo lienzo
    plt.figure()

    # bins=256: una barra para cada valor posible (0-255)
    # range=[0, 256]: forzamos a que el eje X siempre muestre todo el espectro
    # color='black' y alpha=0.7: estética para que se vea pro
    plt.hist(datos_pixel, bins=128, range=[0, 256], color='gray', alpha=0.7)

    # etiquetas
    plt.title(f"Histograma de {img}")
    plt.xlabel("Intensidad (0=Negro, 255=Blanco)")
    plt.ylabel("Frecuencia (Cantidad de Píxeles)")
    plt.grid(True, linestyle='--', alpha=0.5) # cuadricula para leer mejor

    # construir el nombre del archivo
    # os.path.splitext("baboon.tiff") devuelve ("baboon", ".tiff")
    # tomamos el [0] que es el nombre sin extensión
    nombre_limpio = os.path.splitext(img)[0] 
    nombre_hist = f"hist_{nombre_limpio}.png"

    # guardar
    ruta_hist = os.path.join("..", path_img_grayscale, nombre_hist)
    plt.savefig(ruta_hist)
    
    # cerrar la figura
    plt.close()
    
    print(f"\thistograma generado: {nombre_hist}")


path_origen = "../img_grayscale" # donde ya estan las fotos transformadas
path_destino = "../img_tr"       # ruta para guardar las transformaciones
                                 # junto con sus histogramas

if not os.path.exists(path_destino):
    os.makedirs(path_destino)
    print(f"Carpeta creada: {path_destino}")

path_funciones = os.path.join("..", "img_fun")

if not os.path.exists(path_funciones):
    os.makedirs(path_funciones)
    print(f"Carpeta creada: {path_funciones}")

# las LUTS a usar
# buen uso de inspect, lee todas las funciones de transformaciones.py
# las coloca como funciones en esta lista
lista_funciones = [
    obj for name, obj in inspect.getmembers(tr) 
    if inspect.isfunction(obj) and obj.__module__ == tr.__name__
]

for img in imgs:
    # filtramos para leer solo imagenes
    if not img.endswith((".png", ".tiff", ".jpg", ".jpeg")):
        continue
    
    # si el archivo empieza con "hist_", es un histograma viejo, se ignora
    if img.startswith("hist_"):
        continue

    print(f"\nProcesando imagen: {img}")
    
    # leer imagen
    ruta_img = os.path.join(path_origen, img)
    img_gris = cv2.imread(ruta_img)

    L = 255
    
    # aplicando cada una de las funciones de transformacion
    for funcion in lista_funciones:
        nombre_func = funcion.__name__
        
        # crear LUT Vectorizada (0 a 255)
        valores = np.arange(L + 1)
        # aqui llamamos a la funcion dinamicamente
        lut = funcion(valores, L) 
        lut = np.clip(lut, 0, L).astype(np.uint8)
        
        # aplicar LUT
        img_tr = lut[img_gris]
        
        # guardar imagen transformada
        nombre_base = os.path.splitext(img)[0]
        nombre_salida = f"{nombre_base}_{nombre_func}.png"
        cv2.imwrite(os.path.join(path_destino, nombre_salida), img_tr)
        
        # generar y guardar histograma
        plt.figure()
        plt.hist(img_tr.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
        plt.title(f"{nombre_base} - {nombre_func}")
        plt.xlabel("Niveles")
        plt.ylabel("Pixeles")
        
        nombre_hist = f"hist_{nombre_base}_{nombre_func}.png"
        plt.savefig(os.path.join(path_destino, nombre_hist))
        plt.close()

        # ahora crear las imagenes de las funciones
        plt.figure()

        plt.scatter(valores, lut)

        plt.xlabel("Niveles entrada")
        plt.ylabel("Niveles salida")
        plt.title(f"Funcion {funcion.__name__}")
        plt.grid(True, linestyle='--', alpha=0.5) # cuadricula para leer mejor

        plt.savefig(os.path.join(path_funciones, funcion.__name__))
        plt.close()
        
        print(f"\t{nombre_func} listo.")



