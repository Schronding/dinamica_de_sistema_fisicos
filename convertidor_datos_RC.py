import pandas as pd

def analizar_ciclos_rc(nombre_archivo, columna_tiempo, columna_voltaje, umbral_bajo, umbral_inicio_carga):
    """
    Analiza los datos de un experimento RC desde un archivo Excel para encontrar el inicio de los ciclos de carga.

    Args:
        nombre_archivo (str): El nombre del archivo .xlsx a leer.
        columna_tiempo (str): El nombre de la columna que contiene el tiempo.
        columna_voltaje (str): El nombre de la columna que contiene el voltaje.
        umbral_bajo (float): El umbral de voltaje para considerar el capacitor como descargado.
        umbral_inicio_carga (float): El umbral de voltaje para detectar el inicio de una carga.

    Returns:
        list: Una lista con los tiempos de inicio de cada ciclo de carga detectado.
    """
    try:
        # Lee el archivo de Excel y lo carga en un DataFrame de pandas
        df = pd.read_excel(nombre_archivo)
        print(f"Archivo '{nombre_archivo}' leído correctamente.")
        print("Columnas encontradas:", df.columns.tolist())

        # Verifica si las columnas especificadas existen
        if columna_tiempo not in df.columns or columna_voltaje not in df.columns:
            print(f"Error: Asegúrate de que los nombres de las columnas '{columna_tiempo}' y '{columna_voltaje}' sean correctos.")
            return []

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'. Asegúrate de que esté en la misma carpeta que el script.")
        return []
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return []

    # Lista para guardar los tiempos de inicio de cada ciclo
    tiempos_de_inicio = []
    # Estado para saber si el capacitor está descargado y listo para un nuevo ciclo
    capacitor_descargado = True

    # Iteramos sobre cada fila del DataFrame
    for indice, fila in df.iterrows():
        tiempo_actual = fila[columna_tiempo]
        voltaje_actual = fila[columna_voltaje]

        # Condición para detectar el inicio de una carga
        if capacitor_descargado and voltaje_actual > umbral_inicio_carga:
            tiempos_de_inicio.append(tiempo_actual)
            capacitor_descargado = False # El capacitor ha comenzado a cargarse

        # Condición para resetear el estado a "descargado"
        elif not capacitor_descargado and voltaje_actual < umbral_bajo:
            capacitor_descargado = True # El capacitor ya se descargó

    return tiempos_de_inicio

# --- CONFIGURACIÓN ---
# Reemplaza con el nombre de tu archivo de Excel
# NOMBRE_DEL_ARCHIVO = 'tiempoyvoltaje2k14s.xlsx' Para 14 segundos
NOMBRE_DEL_ARCHIVO = 'tiempoyvoltaje1k7s.xlsx' # Para 7 segundos
# Reemplaza con los nombres exactos de tus columnas
# (Simulink usualmente guarda el tiempo en una variable y los datos en otra,
# al exportar a CSV/Excel pueden llamarse 'Time' y 'Data' o los nombres que les diste)
# COLUMNA_TIEMPO = 't (0.01s)'  Para 14 segundos
# COLUMNA_VOLTAJE = 'V'         Para 14 segundos
COLUMNA_TIEMPO = 'Tiempo(0.01s)' # Para 7 segundos
COLUMNA_VOLTAJE = 'V1' # Para 7 segundos 

# Umbrales para detectar los ciclos (puedes ajustarlos)
UMBRAL_BAJO_V = 0.0014       # Voltaje por debajo del cual consideramos el capacitor descargado
UMBRAL_INICIO_CARGA_V = 4.999 # Voltaje que debe superarse para registrar un nuevo ciclo

# --- EJECUCIÓN ---
tiempos_de_carga = analizar_ciclos_rc(
    NOMBRE_DEL_ARCHIVO,
    COLUMNA_TIEMPO,
    COLUMNA_VOLTAJE,
    UMBRAL_BAJO_V,
    UMBRAL_INICIO_CARGA_V
)

# --- RESULTADOS ---
if tiempos_de_carga:
    print("\n--- Tiempos de Inicio de Carga Detectados ---")
    for i, tiempo in enumerate(tiempos_de_carga):
        print(f"Ciclo {i+1}: Inició en el segundo {tiempo:.4f}")
    print(f"\nSe detectaron un total de {len(tiempos_de_carga)} ciclos.")