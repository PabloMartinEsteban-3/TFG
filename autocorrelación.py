import os
import numpy as np

# === CONFIGURACIÓN ===
carpeta_entrada = r'C:\Users\pablo\OneDrive - UNICAN - Estudiantes\Escritorio\Cuarto\TFG\b10_190_splits'
carpeta_salida = 'autocorrelaciones'  # Carpeta donde se guardará el .npy
nombre_archivo = os.path.basename(carpeta_entrada.rstrip('\\/'))  # Usar el nombre de la carpeta como nombre de archivo

# === FUNCIONES ===
def cargar_archivos(carpeta):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.tsv')]
    datos = []
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            data = np.loadtxt(ruta, delimiter='\t')
            datos.append(data)
        except Exception as e:
            print(f"Error cargando {archivo}: {e}")
    return datos

def wrap_angles(phi_array):
    return ((phi_array + np.pi) % (2 * np.pi)) - np.pi

def autocorrelacion_normalizada(signal, max_lag=25):
    signal = np.asarray(signal)
    N = len(signal)
    mu = np.mean(signal)
    var = np.var(signal)
    R = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        num = np.sum((signal[:N - k] - mu) * (signal[k:] - mu))
        denom = (N - k) * var
        R[k] = num / denom
    return np.abs(R)

# === PROCESAMIENTO ===
datos = cargar_archivos(carpeta_entrada)

fases = []
for data in datos:
    x, y = data[:, 1], data[:, 2]
    fase = np.arctan2(y, x)
    fases.append(fase)

fases_concatenadas = np.concatenate(fases)
fases_wrapped = wrap_angles(fases_concatenadas)
Rk = autocorrelacion_normalizada(fases_wrapped, max_lag=25)

# === GUARDAR RESULTADO ===
os.makedirs(carpeta_salida, exist_ok=True)
ruta_salida = os.path.join(carpeta_salida, f'autocorr_{nombre_archivo}.npy')
np.save(ruta_salida, Rk)

print(f'✅ Autocorrelación guardada en: {ruta_salida}')
