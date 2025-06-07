import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Ruta de la carpeta con archivos .npy
carpeta_autocorr = r'C:\Users\pablo\OneDrive - UNICAN - Estudiantes\Escritorio\Cuarto\TFG\autocorrelaciones'  # <-- AJUSTA AQUÍ

# Obtener todos los archivos que empiezan por autocorr_ y terminan en .npy
archivos = [f for f in os.listdir(carpeta_autocorr) if f.startswith('autocorr_') and f.endswith('.npy')]

# Lista de estilos (colores + marcadores) para variar curvas
estilos = [
    {'color': 'blue',    'marker': 'o'},
    {'color': 'green',   'marker': 's'},
    {'color': 'red',     'marker': '^'},
    {'color': 'orange',  'marker': 'v'},
    {'color': 'purple',  'marker': 'D'},
    {'color': 'black',   'marker': '*'},
    {'color': 'brown',   'marker': 'X'},
    {'color': 'teal',    'marker': 'P'},
]

plt.figure(figsize=(10, 6))

# Ordenar archivos por número extraído para que aparezcan ordenados
def extraer_corriente(nombre):
    match = re.search(r'autocorr_b10_(\d+)_splits\.npy', nombre)
    if match:
        return float(match.group(1)) / 10.0  # Ej: 145 → 14.5
    return None

archivos.sort(key=lambda x: extraer_corriente(x) or 0)

# Dibujar todas las autocorrelaciones
for i, archivo in enumerate(archivos):
    corriente = extraer_corriente(archivo)
    if corriente is None:
        print(f"⚠️ No se pudo extraer la corriente de {archivo}")
        continue

    filepath = os.path.join(carpeta_autocorr, archivo)
    datos = np.load(filepath)

    estilo = estilos[i % len(estilos)]  # Ciclar si hay más de 8

    etiqueta = f'I = {str(corriente).replace(".", ",")} mA'
    plt.plot(datos, label=etiqueta, color=estilo['color'],
             marker=estilo['marker'], linestyle='-', linewidth=1)

# Configuración del gráfico
plt.yscale('log')
plt.ylim(1e-5, 1)
plt.xlabel("Retraso $k$", fontsize=14)
plt.ylabel("$|R(k)|$", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
