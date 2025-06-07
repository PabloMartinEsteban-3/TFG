import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta a la carpeta
carpeta = "varianza_programa_pablo_t=20_ns"

# Cargar varianza_t_20ns.csv
ruta_varianza = os.path.join(carpeta, "varianza_t_20ns.csv")
datos_varianza = np.loadtxt(ruta_varianza, delimiter=",", skiprows=1)  # omitir encabezado
tiempo_varianza = datos_varianza[:, 0]
valores_varianza = datos_varianza[:, 1]

# Cargar a_realizaciones.txt
ruta_realizaciones = os.path.join(carpeta, "a_realizaciones.txt")
datos_realizaciones = np.loadtxt(ruta_realizaciones)

# Filtrar hasta t = 20 ns en la primera columna
filtro_20ns = datos_realizaciones[:, 0] <= 20
tiempo_realizaciones = datos_realizaciones[filtro_20ns, 0]
tercera_columna = datos_realizaciones[filtro_20ns, 2]

plt.figure(figsize=(10, 6))
plt.plot(tiempo_varianza, valores_varianza, label='Programa nuevo', color='blue')
plt.plot(tiempo_realizaciones, tercera_columna, label='Programa anterior', color='orange')

plt.xlabel("t (ns)")
plt.ylabel("$\sigma_\phi^2$ (rad^2)")
plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, borderpad=1)
plt.grid(True)
plt.show()
