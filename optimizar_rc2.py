import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# Configuración de archivo CSV
nombre_archivo = 'tiempoyvoltaje1k7s_conpromedio.csv'

# Parámetros ideales del circuito (según práctica No. 1)
R = 1000  # Resistencia [Ω] - 1kOhm para este experimento
C = 1000e-6  # Capacitancia [F] - 1000 microFaradios
V_in = 5.0  # Voltaje de entrada [V] - valor típico

# Lectura de datos del archivo CSV
print(f'Leyendo datos del archivo: {nombre_archivo}')

try:
    # Leer el archivo CSV
    datos = pd.read_csv(nombre_archivo)
    
    # Mostrar los nombres de las columnas disponibles para debugging
    print('Columnas disponibles en el archivo:')
    print(datos.columns.tolist())
    
    # Usar columna 'T' para tiempo y 'Vp' para voltaje promedio
    t_exp = datos['T'].values
    Vc_exp = datos['Vp'].values
    
    print('Datos leídos exitosamente:')
    print(f' - Número de puntos: {len(t_exp)}')
    print(f' - Tiempo: {min(t_exp):.2f} a {max(t_exp):.2f} s')
    print(f' - Voltaje: {min(Vc_exp):.2f} a {max(Vc_exp):.2f} V')
    
except Exception as e:
    print(f'Error al leer el archivo CSV: {e}')
    exit()

# Verificar que los vectores tengan la misma longitud
if len(t_exp) != len(Vc_exp):
    print('Error: Los vectores de tiempo y voltaje deben tener la misma longitud')
    exit()

# Función para calcular R²
def calcular_R2(y_real, y_pred):
    """Calcula el coeficiente de determinación R²"""
    SS_res = np.sum((y_real - y_pred)**2)
    SS_tot = np.sum((y_real - np.mean(y_real))**2)
    if SS_tot == 0:
        return 1
    else:
        return 1 - (SS_res / SS_tot)

# Modelo RC de orden fraccionario
def modelo_rc_fraccionario(x, t, R, C, V_in):
    """
    Modelo RC de orden fraccionario
    x = [α, β, σ]
    """
    alpha = x[0]
    beta = x[1]
    sigma = x[2]
    
    # Evitar división por cero en alpha
    if alpha == 0:
        alpha = 1e-6
    
    # Término del exponente
    exponente = (gamma(beta + 1) / (R * C * alpha)) * (t**alpha) * (sigma**(1 - alpha))
    
    # Modelo completo
    Vc = V_in * (1 - np.exp(-exponente))
    return Vc

# Función para calcular el RMSE
def rmse_circuito_fraccionario(x, t_exp, Vc_exp, R, C, V_in):
    """Función para calcular el RMSE entre el modelo y los datos experimentales"""
    Vc_modelo = modelo_rc_fraccionario(x, t_exp, R, C, V_in)
    
    # Calcular error cuadrático medio (RMSE)
    rmse = np.sqrt(np.mean((Vc_exp - Vc_modelo)**2))
    return rmse

# Configuración de la Optimización
# Parámetros iniciales [α, β, σ]
x0 = [0.5, 0.5, 1.0]  # Valores iniciales para α, β, σ

# Límites de los parámetros [α, β, σ]
bounds = [(0, 0.999), (0, 0.999), (0.1, 10)]  # Límites para α, β, σ

# Opciones del algoritmo de optimización
options = {'maxiter': 1000, 'disp': True}

print('\nIniciando optimización de parámetros...')
print(f'Parámetros iniciales: α={x0[0]:.3f}, β={x0[1]:.3f}, σ={x0[2]:.3f}')

# Función objetivo para la optimización
def objetivo(x):
    return rmse_circuito_fraccionario(x, t_exp, Vc_exp, R, C, V_in)

# Optimización de parámetros
resultado = minimize(objetivo, x0, method='L-BFGS-B', bounds=bounds, options=options)

x_opt = resultado.x
fval = resultado.fun

# Resultados
print('\n--- RESULTADOS DE LA OPTIMIZACIÓN ---')
print('Parámetros óptimos:')
print(f'α = {x_opt[0]:.4f}')
print(f'β = {x_opt[1]:.4f}')
print(f'σ = {x_opt[2]:.4f}')
print(f'RMSE mínimo: {fval:.6f} V')

# Calcular la curva óptima y métricas adicionales
Vc_opt = modelo_rc_fraccionario(x_opt, t_exp, R, C, V_in)
error_max = np.max(np.abs(Vc_exp - Vc_opt))
print(f'Error máximo: {error_max:.6f} V')

error_abs = np.abs(Vc_exp - Vc_opt)
error_rel = error_abs / Vc_exp * 100
error_rel[Vc_exp == 0] = 0  # Evitar división por cero

print(f'Error relativo promedio: {np.mean(error_rel):.2f}%')
print(f'Coeficiente de determinación (R²): {calcular_R2(Vc_exp, Vc_opt):.4f}')

# Crear y mostrar gráfica
print('\nCreando gráfica...')

# Crear figura
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Comparación principal
plt.subplot(2, 3, (1, 2))
plt.plot(t_exp, Vc_exp, 'ro', markersize=4, linewidth=1.5, label='Datos Experimentales')
plt.plot(t_exp, Vc_opt, 'b-', linewidth=2, label='Modelo Ajustado')
plt.xlabel('Tiempo (s)', fontsize=11)
plt.ylabel('Voltaje del Capacitor V_c(t) (V)', fontsize=11)
plt.title('Ajuste del Modelo RC Fraccionario - Comparación', fontsize=12)
plt.legend(loc='southeast', fontsize=10)
plt.grid(True)

# Añadir texto con parámetros óptimos en la gráfica
param_text = f'Parámetros óptimos:\nα = {x_opt[0]:.4f}\nβ = {x_opt[1]:.4f}\nσ = {x_opt[2]:.4f}\nRMSE = {fval:.4f} V'
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
         verticalalignment='top', fontsize=9)

# Subplot 2: Error absoluto
plt.subplot(2, 3, 3)
plt.plot(t_exp, error_abs, 'g-s', linewidth=1.5, markersize=3)
plt.xlabel('Tiempo (s)', fontsize=11)
plt.ylabel('Error Absoluto (V)', fontsize=11)
plt.title('Error Absoluto vs Tiempo', fontsize=11)
plt.grid(True)

# Subplot 3: Error relativo
plt.subplot(2, 3, 4)
plt.plot(t_exp, error_rel, 'm-d', linewidth=1.5, markersize=3)
plt.xlabel('Tiempo (s)', fontsize=11)
plt.ylabel('Error Relativo (%)', fontsize=11)
plt.title('Error Relativo vs Tiempo', fontsize=11)
plt.grid(True)

# Subplot 4: Diagrama de dispersión (Predicho vs Experimental)
plt.subplot(2, 3, 5)
plt.plot(Vc_exp, Vc_opt, 'bo', markersize=4)
plt.plot([min(Vc_exp), max(Vc_exp)], [min(Vc_exp), max(Vc_exp)], 'r--', linewidth=2)
plt.xlabel('Voltaje Experimental (V)', fontsize=11)
plt.ylabel('Voltaje Predicho (V)', fontsize=11)
plt.title('Predicho vs Experimental', fontsize=11)
plt.grid(True)
plt.axis('equal')

# Añadir R² en el gráfico de dispersión
R2 = calcular_R2(Vc_exp, Vc_opt)
plt.text(0.1, 0.9*max(Vc_opt), f'R² = {R2:.4f}', fontsize=11, color='red')

# Subplot 5: Residuales
plt.subplot(2, 3, 6)
residuales = Vc_exp - Vc_opt
plt.plot(t_exp, residuales, 'k-o', linewidth=1.5, markersize=3)
plt.xlabel('Tiempo (s)', fontsize=11)
plt.ylabel('Residuales (V)', fontsize=11)
plt.title('Residuales del Ajuste', fontsize=11)
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)  # Línea en cero

plt.tight_layout()
plt.show()

# Guardar la gráfica
nombre_grafica = 'ajuste_modelo_rc_fraccionario.png'
plt.savefig(nombre_grafica, dpi=300, bbox_inches='tight')
print(f'Gráfica guardada como: {nombre_grafica}')

# Guardar resultados en archivo
resultados = pd.DataFrame({
    'Tiempo': t_exp,
    'Experimental': Vc_exp,
    'Modelo': Vc_opt,
    'Error_Absoluto': error_abs,
    'Error_Relativo': error_rel
})
resultados.to_csv('resultados_ajuste.csv', index=False)
print('Resultados detallados guardados en: resultados_ajuste.csv')

print('\nProceso completado exitosamente!')