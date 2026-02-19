import numpy as np

# Todas las funciones reciben x y L
# x es un arreglo
# L es el valor maximo

def identidad(x, L):
    return x

def negativo(x, L):
    return L - x

def umbral_binario(x, L):
    # regresa L (blanco) si pasa de la mitad, 0 (negro) si no
    # np.where es como un "if" para arreglos
    return np.where(x > L/2, L, 0)

def gamma(x, L, g=0.5):
    # formula: L * (x/L)^g
    return L * np.power(x / L, g)

def logaritmica(x, L):
    # formula: c * log(1 + x)
    c = L / np.log(1 + L)
    return c * np.log(1 + x)

def sigmoide(x, L, s=15):
    centro = L / 2
    return L / (1 + np.exp(-s * (x - centro) / L))
