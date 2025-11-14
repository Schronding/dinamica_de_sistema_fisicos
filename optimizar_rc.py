"""
Código para Ajuste de Parámetros de Circuito RC Fraccionario
Traducción a Python de un script de MATLAB.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma  # Para la función Gamma (Γ)
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# --- 1. Configuración de archivo y Parámetros (MODIFICAR ESTA SECCIÓN) ---
# =========================================================================
# Nombre del archivo CSV (debe estar en la misma carpeta que este script)
# Puedes usar 'datos_exp_2k.csv' o 'tiempoyvoltaje1k7s_conpromedio.csv'
NOMBRE_ARCHIVO = 'datos_exp_2k.csv'

# Parámetros ideales/medidos del circuito
R = 2140.0      # Resistencia [Ω] (Usando el valor medido de tu compañero)
C = 1000e-6     # Capacitancia [F] (1000 µF)
V_IN = 4.99     # Voltaje de entrada [V] (Usando el valor medido)
TIME_STEP = 0.01 # Paso de tiempo de tus datos (0.01s)
# =========================================================================

def modelo_rc_fraccionario(x, t, R, C, V_in):
    """
    Define el modelo RC de orden fraccionario.
    x = [α, β, σ]
    """
    alpha, beta, sigma = x
    
    # Evitar división por cero en alpha
    if alpha == 0:
        alpha = 1e-6
    
    # Término del exponente
    # Nota: Usamos gamma(beta + 1) que es Γ(β+1)
    exponente = (gamma(beta + 1) / (R * C * alpha)) * (t**alpha) * (sigma**(1 - alpha))
    
    # Modelo completo
    Vc = V_in * (1 - np.exp(-exponente))
    return Vc

def funcion_costo_rmse(x, t_exp, Vc_exp, R, C, V_in):
    """
    Función objetivo: Calcula el RMSE entre el modelo y los datos.
    """
    # Predecir voltaje con el modelo
    Vc_modelo = modelo_rc_fraccionario(x, t_exp, Vc_exp, R, C, V_in)
    
    # Calcular RMSE
    # Usamos np.nan_to_num para evitar errores si hay datos faltantes (NaN)
    rmse = np.sqrt(mean_squared_error(np.nan_to_num(Vc_exp), np.nan_to_num(Vc_modelo)))
    return rmse

def calcular_r2(y_real, y_pred):
    """Calcula el coeficiente de determinación R²."""
    # Quitar NaNs para un cálculo robusto
    mask = ~np.isnan(y_real) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan # No hay datos válidos
    return r2_score(y_real[mask], y_pred[mask])

def main():
    """Función principal para ejecutar el análisis."""
    print(f"Leyendo datos del archivo: {NOMBRE_ARCHIVO}")

    # --- 2. Lectura de datos del archivo CSV ---
    try:
        datos = pd.read_csv(NOMBRE_ARCHIVO)
        print("Columnas disponibles:", datos.columns.tolist())
        
        # Lógica para encontrar columnas de tiempo y voltaje
        if 'Tiempoex' in datos.columns:
            t_exp_raw = datos['Tiempoex']
            print("Usando columna 'Tiempoex' para tiempo")
        elif 'tiempoex' in datos.columns:
            t_exp_raw = datos['tiempoex']
            print("Usando columna 'tiempoex' para tiempo")
        elif 'T' in datos.columns:
            # Asumimos que T es un índice que empieza en 1
            t_exp_raw = (datos['T'] - 1) * TIME_STEP
            print(f"Usando columna 'T' y recalculando tiempo con paso {TIME_STEP}s")
        else:
            t_exp_raw = datos.iloc[:, 0]
            print("Usando primera columna para tiempo")
            
        if 'Volt_prom' in datos.columns:
            Vc_exp_raw = datos['Volt_prom']
            print("Usando columna 'Volt_prom' para voltaje")
        elif 'volt_prom' in datos.columns:
            Vc_exp_raw = datos['volt_prom']
            print("Usando columna 'volt_prom' para voltaje")
        elif 'Vp' in datos.columns:
            Vc_exp_raw = datos['Vp']
            print("Usando columna 'Vp' para voltaje")
        else:
            Vc_exp_raw = datos.iloc[:, 1]
            print("Usando segunda columna para voltaje")
        
        # Convertir a arrays de NumPy para cálculos
        t_exp = np.array(t_exp_raw)
        Vc_exp = np.array(Vc_exp_raw)
        
        # Limpiar datos de NaNs (datos faltantes)
        mask = ~np.isnan(t_exp) & ~np.isnan(Vc_exp)
        t_exp = t_exp[mask]
        Vc_exp = Vc_exp[mask]

        print(f"Datos leídos exitosamente:")
        print(f" - Número de puntos: {len(t_exp)}")
        print(f" - Tiempo: {np.min(t_exp):.2f} a {np.max(t_exp):.2f} s")
        print(f" - Voltaje: {np.min(Vc_exp):.2f} a {np.max(Vc_exp):.2f} V")
        
    except FileNotFoundError:
        print(f"*** ERROR: No se encontró el archivo '{NOMBRE_ARCHIVO}' ***")
        return
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    # --- 3. Configuración de la Optimización ---
    # Parámetros iniciales [α, β, σ]
    x0 = [0.5, 0.5, 1.0]
    
    # Límites de los parámetros [(α_min, α_max), (β_min, β_max), (σ_min, σ_max)]
    bounds = [(0, 0.999), (0, 0.999), (0.1, 10)]
    
    # Función de costo (RMSE)
    # Usamos una función lambda para pasar los argumentos adicionales (t, Vc, R, C, V_in)
    cost_function = lambda x: funcion_costo_rmse(x, t_exp, Vc_exp, R, C, V_IN)

    # --- 4. Optimización de Parámetros ---
    print('\nIniciando optimización de parámetros...')
    print(f'Parámetros iniciales: α={x0[0]:.3f}, β={x0[1]:.3f}, σ={x0[2]:.3f}')
    
    # 'minimize' es el equivalente de 'fmincon' en MATLAB
    # Usamos el método 'SLSQP' (Sequential Least Squares Programming) o 'L-BFGS-B'
    # que manejan bien los límites (bounds)
    resultado_opt = minimize(cost_function, 
                             x0, 
                             method='SLSQP', # o 'L-BFGS-B'
                             bounds=bounds, 
                             options={'disp': True, 'maxiter': 1000, 'ftol': 1e-6})

    # --- 5. Resultados ---
    if not resultado_opt.success:
        print(f"*** ADVERTENCIA: La optimización falló o no convergió. ***")
        print(f"Mensaje: {resultado_opt.message}")
        return

    x_opt = resultado_opt.x
    rmse_min = resultado_opt.fun

    print('\n--- RESULTADOS DE LA OPTIMIZACIÓN ---')
    print('Parámetros óptimos:')
    print(f'α = {x_opt[0]:.4f}')
    print(f'β = {x_opt[1]:.4f}')
    print(f'σ = {x_opt[2]:.4f}')
    print(f'RMSE mínimo: {rmse_min:.6f} V')

    # Calcular la curva óptima y métricas adicionales
    Vc_opt = modelo_rc_fraccionario(x_opt, t_exp, R, C, V_IN)
    error_abs = np.abs(Vc_exp - Vc_opt)
    error_rel = error_abs / (Vc_exp + 1e-9) * 100 # +1e-9 para evitar división por cero
    
    r2_val = calcular_r2(Vc_exp, Vc_opt)

    print(f'Error máximo: {np.max(error_abs):.6f} V')
    print(f'Error relativo promedio: {np.mean(error_rel):.2f}%')
    print(f'Coeficiente de determinación (R²): {r2_val:.4f}')

    # --- 6. Crear y mostrar gráfica ---
    print('\nCreando gráfica...')
    fig = plt.figure(figsize=(15, 9))
    plt.suptitle('Ajuste del Modelo RC Fraccionario - Análisis de Resultados', fontsize=16)

    # Subplot 1: Comparación principal
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax1.plot(t_exp, Vc_exp, 'r.', markersize=5, label='Datos Experimentales', alpha=0.6)
    ax1.plot(t_exp, Vc_opt, 'b-', linewidth=2, label='Modelo Ajustado')
    ax1.set_xlabel('Tiempo (s)', fontsize=11)
    ax1.set_ylabel('Voltaje del Capacitor Vc(t) (V)', fontsize=11)
    ax1.set_title('Comparación Principal', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True)

    # Añadir texto con parámetros óptimos
    param_text = (f'Parámetros óptimos:\n'
                  f'α = {x_opt[0]:.4f}\n'
                  f'β = {x_opt[1]:.4f}\n'
                  f'σ = {x_opt[2]:.4f}\n'
                  f'RMSE = {rmse_min:.4f} V\n'
                  f'R² = {r2_val:.4f}')
    ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 2: Error absoluto
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax2.plot(t_exp, error_abs, 'g-')
    ax2.set_xlabel('Tiempo (s)', fontsize=11)
    ax2.set_ylabel('Error Absoluto (V)', fontsize=11)
    ax2.set_title('Error Absoluto vs Tiempo', fontsize=11)
    ax2.grid(True)

    # Subplot 3: Diagrama de dispersión (Predicho vs Experimental)
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    ax3.plot(Vc_exp, Vc_opt, 'b.', alpha=0.5)
    ax3.plot([min(Vc_exp), max(Vc_exp)], [min(Vc_exp), max(Vc_exp)], 'r--', linewidth=2, label='Ideal (Y=X)')
    ax3.set_xlabel('Voltaje Experimental (V)', fontsize=11)
    ax3.set_ylabel('Voltaje Predicho (V)', fontsize=11)
    ax3.set_title('Predicho vs Experimental', fontsize=11)
    ax3.grid(True)
    ax3.legend()
    ax3.axis('equal')

    # Subplot 4: Residuales
    ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    residuales = Vc_exp - Vc_opt
    ax4.plot(t_exp, residuales, 'k.', markersize=4, alpha=0.6)
    ax4.axhline(0, color='r', linestyle='--', linewidth=1)
    ax4.set_xlabel('Tiempo (s)', fontsize=11)
    ax4.set_ylabel('Residuales (V)', fontsize=11)
    ax4.set_title('Residuales del Ajuste', fontsize=11)
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout
    
    # --- 7. Guardar la gráfica y resultados ---
    nombre_grafica = 'ajuste_modelo_rc_fraccionario_PY.png'
    fig.savefig(nombre_grafica)
    print(f"Gráfica guardada como: {nombre_grafica}")

    # Guardar resultados en archivo CSV
    resultados = pd.DataFrame({
        'Tiempo': t_exp,
        'Experimental': Vc_exp,
        'Modelo': Vc_opt,
        'Error_Absoluto': error_abs,
        'Error_Relativo': error_rel
    })
    resultados.to_csv('resultados_ajuste_PY.csv', index=False)
    print(f'Resultados detallados guardados en: resultados_ajuste_PY.csv')

    plt.show() # Mostrar la gráfica

# --- Punto de entrada del script ---
if __name__ == '__main__':
    main()