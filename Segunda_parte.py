import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ========================
# CONFIGURACIÓN GENERAL
# ========================

n_fich = 20_500_000
n_reali = 10_250
n_puntos = int(n_fich / n_reali)
deltat = np.float64(500e-12)
window = deltat * n_puntos
pi = np.float64(np.pi)
tiempo_comun = np.linspace(0, 1000, n_puntos)  # en nanosegundos

carpeta = r'C:\Users\pablo\OneDrive - UNICAN - Estudiantes\Escritorio\Cuarto\TFG\b10_145_splits'

# ========================
# FUNCIONES AUXILIARES
# ========================

def seleccionar_archivos_aleatorios(carpeta, num_archivos=5):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.tsv') and os.path.isfile(os.path.join(carpeta, f))]
    return [os.path.join(carpeta, archivo) for archivo in random.sample(archivos, num_archivos)]

def cargar_archivos(archivos):
    datos = []
    for archivo in archivos:
        try:
            data = np.loadtxt(archivo, delimiter='\t')
            datos.append(data)
            print(f"Cargado: {archivo} con {data.shape} datos")
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")
    return datos

def cargar_todos_los_archivos(carpeta):
    archivos = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.endswith('.tsv')]
    return cargar_archivos(archivos)

# ========================
# CARGA DE DATOS
# ========================

archivos_seleccionados = seleccionar_archivos_aleatorios(carpeta)
muestra_datos = cargar_archivos(archivos_seleccionados)
todos_los_datos = cargar_todos_los_archivos(carpeta)

# ========================
# HISTOGRAMAS I y Q
# ========================

def graficar_histogramas_globales(datos, sigma=2, bins=400):
    I_total = np.concatenate([d[:, 1] for d in datos])
    Q_total = np.concatenate([d[:, 2] for d in datos])

    I_total /= np.max(np.abs(I_total))
    Q_total /= np.max(np.abs(Q_total))
    rango = (-1, 1)

    for señal, color, label in zip([I_total, Q_total], ['blue', 'green'], ['p(I)', 'p(Q)']):
        hist, edges = np.histogram(señal, bins=bins, range=rango, density=True)
        centros = 0.5 * (edges[:-1] + edges[1:])
        hist_suave = gaussian_filter1d(hist, sigma=sigma)

        plt.figure(figsize=(8, 4))
        plt.plot(centros, hist_suave, color=color, linewidth=2)
        plt.ylim(0, 2)
        plt.xlim(-1, 1)
        plt.xlabel('Voltaje (V)', fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

graficar_histogramas_globales(todos_los_datos)

# ========================
# FASE Y UNWRAP
# ========================

# Calcular la fase del láser para cada archivo cargado
fases_muestra = []
fases_totales = []
fases_unwrapp = []

for data in muestra_datos:
    x = data[:, 1]  # Segunda columna
    y = data[:, 2]  # Tercera columna
    fase = np.arctan2(y, x)  # Calcular la fase
    fase_unwrapped = np.unwrap(fase)
    fases_muestra.append(fase_unwrapped)

for data in todos_los_datos:
    x2 = data[:, 1]  # Segunda columna
    y2 = data[:, 2]  # Tercera columna
    fase = np.arctan2(y2, x2)  # Calcular la fase
    fase_unwrapped = np.unwrap(fase)
    fases_unwrapp.append(fase_unwrapped)
    fases_totales.append(fase)

Fases_unwrapp_totales = np.concatenate(fases_unwrapp)  # Concatenar todas las fases desenvueltas
fases_totales_array = np.concatenate(fases_totales)


# Se grafican las fases de los 5 archivos aleatorios
plt.figure(figsize=(10, 6))
for i, fase in enumerate(fases_muestra):
    plt.plot(tiempo_comun, fase, label=f'Archivo {i+1}', linewidth=2)

plt.xlabel('t (ns)', fontsize=14)
plt.ylabel(r'$\theta$ (rad)', fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()



# ========================
# WRAPPING + HISTOGRAMA
# ========================

def wrap_angles(phi_array):
    wrapped_array = np.zeros_like(phi_array)
    for i, phi in enumerate(phi_array):
        k = int(phi / pi)
        if k % 2 == 0:
            wrapped_array[i] = phi - pi * k
        elif k > 0:
            wrapped_array[i] = phi - pi * k - pi
        else:
            wrapped_array[i] = phi - pi * k + pi
    return wrapped_array

fases_wrapped = wrap_angles(fases_totales_array)

plt.figure(figsize=(8, 5))
plt.hist(fases_wrapped, bins=1024, range=(-pi, pi), edgecolor='black', alpha=0.7, density=True)
plt.xlabel(r'$\theta$ (rad)', fontsize=14)
plt.ylabel(r'p($\theta$)', fontsize=14)
plt.xticks([-pi, -pi/2, 0, pi/2, pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.yticks([0, 1 / (2 * pi)], ['0', r'1/2$\pi$'])
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ========================
# HISTOGRAMA SUAVIZADO DE FASE
# ========================

def graficar_histograma_suavizado_fase_normalizada(fases_wrapped, sigma=1.2, bins=400):
    fases_norm = fases_wrapped / pi
    hist, edges = np.histogram(fases_norm, bins=bins, range=(-1, 1), density=True)
    centros = 0.5 * (edges[:-1] + edges[1:])
    hist_suave = gaussian_filter1d(hist, sigma=sigma)

    plt.figure(figsize=(8, 4))
    plt.plot(centros, hist_suave, color='purple', linewidth=2)
    plt.ylim(0, 1.25)
    plt.xlim(-1, 1)
    plt.xlabel(r'Fase($\theta$)', fontsize=12)
    plt.ylabel(r'p($\theta$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

graficar_histograma_suavizado_fase_normalizada(fases_wrapped)

# ========================
# MÉTRICAS DE ENTROPÍA
# ========================

hist, bin_edges = np.histogram(fases_wrapped, bins=1024, range=(-pi, pi), density=True)
bin_width = np.diff(bin_edges)[0]
probabilidades = hist * bin_width
entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-10))
entropia_maxima = np.log2(len(hist))
porcentaje_aleatoriedad = (entropia / entropia_maxima) * 100
p_max = np.max(probabilidades)
hmin = -np.log2(p_max)

print(f"Min-Entropy (Hmin): {hmin:.4f} bits")
print(f"Entropía: {entropia:.4f} / Máxima: {entropia_maxima:.4f}")
print(f"Porcentaje de aleatoriedad: {porcentaje_aleatoriedad:.2f}%")

# ========================
# HISTOGRAMA DE FASES DESENVUELTAS
# ========================

plt.figure(figsize=(8, 5))
plt.hist(Fases_unwrapp_totales, bins=300, edgecolor='black', alpha=0.7, density=True)
plt.xlabel(r'$\theta$ (rad)', fontsize=14)
plt.ylabel(r'p($\theta$)', fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ========================
# AUTOCORRELACIÓN
# ========================

def autocorrelacion_normalizada(signal, max_lag=25):
    signal = np.asarray(signal)
    N = len(signal)
    mu, var = np.mean(signal), np.var(signal)
    R = np.zeros(max_lag + 1)

    for k in range(max_lag + 1):
        R[k] = np.sum((signal[:N-k] - mu) * (signal[k:] - mu)) / ((N - k) * var)
    return R

Rk = np.abs(autocorrelacion_normalizada(fases_wrapped, max_lag=25))

plt.figure(figsize=(8, 5))
plt.plot(Rk, marker='o')
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.xlabel("Retraso $k$", fontsize=14)
plt.ylabel("$|R(k)|$", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# EXTRACCIÓN VON NEUMANN
# ========================

def von_neumann_extraction(values, bits=10):
    output_numbers = []
    raw_bits = []  # Lista para almacenar los bits extraídos con Von Neumann

    print(f"[INFO] Número total de valores de entrada: {len(values)}")

    for i, value in enumerate(values):
        y = (value + np.pi) / (2 * np.pi)
        n = int(y * (2 ** bits))
        bin_str = format(n, f'0{bits}b')

        # Aplicar Von Neumann extractor a cada binario individual
        for j in range(0, len(bin_str) - 1, 2):
            pair = bin_str[j:j+2]
            if pair == '01':
                raw_bits.append('0')
            elif pair == '10':
                raw_bits.append('1')
            # 00 y 11 se descartan

        if i < 5:
            print(f"[DEBUG] Valor {i}: {value:.4f} → bin={bin_str} → bits extraídos: {''.join(raw_bits[-5:])}")

    total_bits = len(raw_bits)
    total_blocks = total_bits // bits

    print(f"[INFO] Total de bits extraídos: {total_bits}")
    print(f"[INFO] Se podrán formar {total_blocks} bloques de {bits} bits")

    for i in range(0, total_bits - bits + 1, bits):
        bloque = ''.join(raw_bits[i:i+bits])
        dec_value = int(bloque, 2)
        normalized = dec_value / (2 ** bits)
        output_numbers.append(normalized)

        if i < 5 * bits:
            print(f"[DEBUG] Bloque {i//bits + 1}: {bloque} → {dec_value} → {normalized:.4f}")

    print(f"[INFO] Total de valores normalizados generados: {len(output_numbers)}")

    return output_numbers


results = von_neumann_extraction(fases_wrapped, bits=10)
print("Números normalizados extraídos:", results)



# Histograma
plt.hist(results, bins=50, edgecolor='black', alpha=0.7, density=True)
#plt.title("Distribución de los números extraídos")
plt.xlabel(r'$\theta$ (rad)', fontsize=14)
plt.ylabel(r'p($\theta$)', fontsize=14)
plt.yticks([0,1], labels=['0', '1'])
plt.xticks([0, 1],labels=['0', '1'])
plt.grid(True)
plt.show()
