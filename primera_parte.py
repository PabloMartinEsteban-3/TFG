import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Parámetros de adquisición ---
n_total_datos = 20_500_000            # Total de puntos en el archivo
n_realizaciones = 10_250              # Número de realizaciones
n_puntos = n_total_datos // n_realizaciones  # Puntos por realización
deltat = np.float64(500e-12)          # Intervalo de muestreo (500 ps)
window = deltat * n_puntos            # Duración de cada realizaciónç
pi = np.float64(np.pi)

# --- Eje de tiempo común para graficar (en nanosegundos) ---
tiempo_comun = np.linspace(0, 1000, n_puntos)  # ns

def seleccionar_archivos_aleatorios(carpeta, num_archivos=5):
    """Selecciona rutas completas de archivos .tsv aleatorios dentro de una carpeta."""
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.tsv') and os.path.isfile(os.path.join(carpeta, f))]
    seleccionados = random.sample(archivos, num_archivos)
    return [os.path.join(carpeta, archivo) for archivo in seleccionados]

def cargar_archivos(rutas_archivos):
    """Carga archivos TSV como arrays NumPy."""
    datos = []
    for archivo in rutas_archivos:
        try:
            data = np.loadtxt(archivo, delimiter='\t')
            datos.append(data)
            print(f"Cargado: {archivo} con {data.shape} datos")
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")
    return datos

def cargar_todos_los_archivos(carpeta):
    """Carga todos los archivos .tsv de una carpeta."""
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.tsv') and os.path.isfile(os.path.join(carpeta, f))]
    rutas_completas = [os.path.join(carpeta, f) for f in archivos]
    return cargar_archivos(rutas_completas)

# Carpeta que contiene los datos
carpeta = r'C:\Users\pablo\OneDrive - UNICAN - Estudiantes\Escritorio\Cuarto\TFG\q5_splits2'

# Selección y carga de 5 archivos aleatorios
archivos_aleatorios = seleccionar_archivos_aleatorios(carpeta)
muestra_datos = cargar_archivos(archivos_aleatorios)

# Carga de todos los archivos disponibles
todos_los_datos = cargar_todos_los_archivos(carpeta)

# =========================
# 1. Calculo de la fase desenrollada
# =========================

fases_muestra = []
for datos in muestra_datos:
    x = datos[:, 1]  # Columna X
    y = datos[:, 2]  # Columna Y
    fase = np.arctan2(y, x)
    fase_unwrapped = np.unwrap(fase)
    fases_muestra.append(fase_unwrapped)

# Gráfico de fases
plt.figure(figsize=(10, 6))
for fase in fases_muestra:
    plt.plot(tiempo_comun, fase)

plt.xlabel('t (ns)')
plt.ylabel(r'$\theta$ (rad)')
plt.title(r'$\theta$ para 5 archivos aleatorios')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os
import random

# =========================
# 2. Derivada Temporal de la Fase
# =========================

def calcular_derivada_temporal(fase, deltat):
    """
    Calcula la derivada temporal de la fase normalizada por 2π.
    """
    derivada = np.zeros_like(fase)
    for j in range(1, len(fase)):
        derivada[j] = (fase[j] - fase[j - 1]) / deltat / (2 * np.pi)
    return derivada

# Graficar derivada de la fase para cada muestra
plt.figure(figsize=(10, 6))
for fase in fases_muestra:
    derivada = calcular_derivada_temporal(fase, deltat)
    plt.plot(tiempo_comun, derivada)

plt.xlabel('t (ns)')
plt.ylabel(r'$\frac{1}{2\pi} \frac{d\phi}{dt}$ (GHz)')
plt.title('Derivada Temporal de la Fase')
plt.show()

# =========================
# 3. Cálculo de Delta_w
# =========================

def calcular_delta_w(datos, deltat):
    """
    Calcula Δω para cada archivo a partir de la fase desenvuelta.
    """
    delta_ws = []

    for i, data in enumerate(datos):
        x = data[:, 1]
        y = data[:, 2]
        fase = np.arctan2(y, x)
        fase_unwrapped = np.unwrap(fase)

        N = len(fase_unwrapped)
        if N < 2:
            print(f"Archivo {i} tiene muy pocos datos. Saltando...")
            continue

        sumatorio = fase_unwrapped[-1] - fase_unwrapped[0]
        delta_w = (1 / (deltat * (N - 1))) * sumatorio / (2 * np.pi)
        delta_ws.append(delta_w)

    return delta_ws

# Calcular Δω y preparar datos
delta_ws = calcular_delta_w(todos_los_datos, deltat)
delta_ws_MHz = np.array(delta_ws) / 1e6
tiempo = np.arange(len(delta_ws)) * window  # Convertir a nanosegundos
tiempo_microsegundos = tiempo * 1e6
media_delta_w = np.mean(delta_ws)
desviacion_delta_w = np.std(delta_ws, ddof=1)

# Mostrar resultados
print(f"Numero de archivos: {len(delta_ws)}")
print(f"Valor medio de Δω: {media_delta_w:.6f} Hz")
print(f"Desviación estándar de Δω: {desviacion_delta_w:.6f} Hz")

# Graficar Δω en MHz
plt.plot(tiempo_microsegundos, delta_ws_MHz, marker='o', markersize=4,
         linestyle='None', color='black')
plt.xlim(0, 1000)
plt.xlabel('t (µs)')
plt.ylabel(r'$\frac{1}{2\pi} \Delta W$ (MHz)')
plt.title(r'Valores de $\frac{1}{2\pi} \Delta W$ en MHz para cada archivo')
plt.grid(True)
plt.show()

# =========================
# 4. Fase Corregida
# =========================

tiempo_seg = tiempo_comun * 1e-9
delta_ws_rad = np.array(delta_ws) * (2 * np.pi)

def fases_phi(datos, tiempo, delta_ws):
    """
    Corrige la fase para cada archivo restando la tendencia lineal.
    """
    fase_phi = []

    for i, data in enumerate(datos):
        x = data[:, 1]
        y = data[:, 2]
        fase = np.arctan2(y, x)
        fase_unwrapped = np.unwrap(fase)

        N = len(fase_unwrapped)
        if N < 2:
            print(f"Archivo {i} tiene muy pocos datos. Saltando...")
            continue

        fase_corregida = [
            fase_unwrapped[j] - (delta_ws_rad[i] * tiempo_seg[j])
            for j in range(N)
        ]
        fase_phi.append(fase_corregida)

    return fase_phi

fase_phi_resultado = fases_phi(todos_los_datos, tiempo_comun, delta_ws)

# Selección aleatoria de hasta 5 muestras
num_muestras = min(5, len(fase_phi_resultado))
muestras_seleccionadas = random.sample(fase_phi_resultado, num_muestras)

# Graficar fases corregidas
plt.figure(figsize=(10, 6))
for fase in muestras_seleccionadas:
    plt.plot(tiempo_comun, fase)
plt.xlabel('t (ns)')
plt.ylabel(r'$\phi$ (rad)')
plt.title(r'Fase corregida $\phi$ para 5 archivos aleatorios')
plt.grid(True)
plt.show()

# =========================
# 5. Fase Media
# =========================

def calcular_fase_media(fase_phi):
    return np.mean(np.array(fase_phi), axis=0)

fase_media_resultado = calcular_fase_media(fase_phi_resultado)

plt.figure(figsize=(10, 6))
plt.plot(tiempo_comun, fase_media_resultado, label="Fase Media", color="red")
plt.ylim(-3, 3)
plt.xlabel('t (ns)')
plt.ylabel(r'<$\varphi$> (rad)')
plt.title(r'Fase media $\phi$ de todos los archivos')
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 6. Desviacion estandar
# =========================

def calcular_desviacion_estandar(fase_phi):
    return np.std(np.array(fase_phi), axis=0)

desviacion_resultado = calcular_desviacion_estandar(fase_phi_resultado)

plt.figure(figsize=(10, 6))
plt.plot(tiempo_comun, desviacion_resultado, label="Desviación estándar", color="blue")
plt.ylim(0, np.max(desviacion_resultado) * 1.1)
plt.xlabel('t (ns)')
plt.ylabel(r'$\sigma_{\varphi}$ (rad)')
plt.title(r't (ns)')
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 7. Varianza de la Fase
# =========================

def calcular_varianza_fase_phi(fase_phi):
    fase_phi_array = np.array(fase_phi)
    return np.mean(fase_phi_array**2, axis=0) - np.mean(fase_phi_array, axis=0)**2

varianza_fase_phi = calcular_varianza_fase_phi(fase_phi_resultado)
indices_hasta_20ns = tiempo_comun <= 20
tiempo_filtrado = tiempo_comun[indices_hasta_20ns]
varianza_filtrada = varianza_fase_phi[indices_hasta_20ns]

# Guardar CSV
os.makedirs("varianza_programa_pablo_t=20_ns", exist_ok=True)
np.savetxt("varianza_programa_pablo_t=20_ns/varianza_t_20ns.csv",
           np.column_stack((tiempo_filtrado, varianza_filtrada)),
           delimiter=",", header="Tiempo (ns), Varianza(phi) (rad^2)", comments='')

# Graficar varianza
plt.figure(figsize=(10, 6))
plt.plot(tiempo_comun, varianza_fase_phi,
         label=r'$\langle \phi^2 \rangle - \langle \phi \rangle^2$', color='blue')
plt.xlabel('t (ns)')
plt.ylabel(r'$\sigma_\phi$ (rad$^2$)')
plt.title(r'Varianza de $\phi$ en función del tiempo')
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 8. Fase a los 20 ns
# =========================

def graficar_fase_phi_20ns(muestras_seleccionadas, tiempo_comun):
    idx_inicio = np.searchsorted(tiempo_comun, 0)
    idx_fin = np.searchsorted(tiempo_comun, 20)
    phi_finales = []

    plt.figure(figsize=(10, 6))
    for fase in muestras_seleccionadas:
        fase_recortada = fase[idx_inicio:idx_fin]
        tiempo_recortado = tiempo_comun[idx_inicio:idx_fin]
        plt.plot(tiempo_recortado, fase_recortada)
        phi_finales.append(fase_recortada[-1])

    plt.xlabel('t (ns)')
    plt.ylabel(r'$\phi$ (rad)')
    plt.title(r'Fase corregida $\phi$ de 0 a 20 ns para 5 archivos aleatorios')
    plt.grid(True)
    plt.show()
    return phi_finales

phi_20ns_muestras = graficar_fase_phi_20ns(muestras_seleccionadas, tiempo_comun)

def obtener_phi_20ns(fase_phi, tiempo_comun):
    idx_20ns = np.searchsorted(tiempo_comun, 20)
    return [fase[idx_20ns] for fase in fase_phi]

phi_finales_20ns = obtener_phi_20ns(fase_phi_resultado, tiempo_comun)

# =========================
# 9. Histograma y gaussiana
# =========================


def graficar_histograma_phi_20ns(phi_finales):
    # Calcular media y desviación estándar del conjunto
    mu = np.mean(phi_finales)
    sigma = np.std(phi_finales, ddof=1)

    # Histograma
    plt.figure(figsize=(10, 6))
    count, bins, _ = plt.hist(phi_finales, bins=40, color='blue', edgecolor='black', alpha=0.8, density=True, label='Histograma')

    # Valores para graficar la gaussiana
    x = np.linspace(min(phi_finales), max(phi_finales), 500)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, color='red', linewidth=2,
             label=fr'Gaussiana ajustada: $\mu$ = {mu:.2f} rad, $\sigma$ = {sigma:.2f} rad')

    # Estética
    plt.xlabel(r'$\phi$ en 20 ns (rad)')
    plt.ylabel('fdp')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


graficar_histograma_phi_20ns(phi_finales_20ns)


# =========================
# 10. Envuelto de Ángulos
# =========================

def wrap_angles(phi_array):
    """
    Envuelve los ángulos a [-π, π] con ifs.
    """
    wrapped_array = np.zeros_like(phi_array)
    for i, phi in enumerate(phi_array):
        k = int(phi / np.pi)
        if k % 2 == 0:
            wrapped_array[i] = phi - np.pi * k
        elif k > 0:
            wrapped_array[i] = phi - np.pi * k - np.pi
        else:
            wrapped_array[i] = phi - np.pi * k + np.pi
    return wrapped_array

wrapped_angles = wrap_angles(phi_finales_20ns)

plt.figure(figsize=(8, 5))
plt.hist(wrapped_angles, bins=30, range=(-np.pi, np.pi),
         edgecolor='black', alpha=0.7, density=True)
plt.xlabel("fase en 20 ns (rad)")
plt.ylabel(r'p($\theta$)')
plt.yticks([0, 1/(2*np.pi)], labels=['0', r'1/2$\pi$'])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           labels=[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# =========================
# 11. Entropía del Histograma
# =========================

def wrap_cero_a_uno(wrapped_array):
    return (wrapped_array + np.pi) / (2 * np.pi)

wrapped_cero_a_uno = wrap_cero_a_uno(wrapped_angles)
hist, _ = np.histogram(wrapped_angles, bins=1024, range=(-np.pi, np.pi), density=True)

probabilidades = hist / np.sum(hist)
entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-10))
entropia_maxima = np.log2(len(hist))
porcentaje_aleatoriedad = (entropia / entropia_maxima) * 100
# Paso 3: calcular Hmin
p_max = np.max(probabilidades)
hmin = -np.log2(p_max)

print(f"p_max: {p_max:.4f}")
print(f"Min-Entropy (Hmin): {hmin:.4f} bits")
print(f"La entropía del histograma es: {entropia:.4f}")
print(f"La entropía máxima esperada es: {entropia_maxima:.4f}")
print(f"Porcentaje de aleatoriedad: {porcentaje_aleatoriedad:.2f}%")

plt.figure(figsize=(8, 5))
plt.hist(wrapped_cero_a_uno, bins=30, range=(0, 1),
         edgecolor='black', alpha=0.7, density=True)
plt.xlabel("fase en 20 ns (rad)")
plt.ylabel("Frecuencia")
plt.title("Histograma de Ángulos Envuelto en [0, 1]")
plt.yticks([0, 1], labels=['0', '1'])
plt.xticks([0, 1], labels=['0', '1'])
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
