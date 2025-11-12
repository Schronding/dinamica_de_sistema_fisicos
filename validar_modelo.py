import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')
TIME_STEP = 0.01

def analizar_y_validar(csv_file, R_valor, time_col, volt_col, es_indice_t=False):
    print(f"\n--- Iniciando Análisis para R = {R_valor}Ω ---")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"*** ERROR: No se pudo encontrar el archivo '{csv_file}' ***")
        print("Asegúrate de que el nombre esté escrito exactamente igual y esté en la misma carpeta.")
        return
    except Exception as e:
        print(f"*** ERROR al leer el archivo: {e} ***")
        return

    if time_col not in df.columns or volt_col not in df.columns:
        print(f"*** ERROR: No se encontraron las columnas '{time_col}' o '{volt_col}' en el archivo. ***")
        print(f"Columnas disponibles: {df.columns.tolist()}")
        return

    if es_indice_t:
        df['Time_s'] = (df[time_col] - 1) * TIME_STEP
        time_col = 'Time_s'
        
    y_exp = df[volt_col]
    t_exp = df[time_col]
    
    print("Datos cargados exitosamente.")

    V_final = y_exp.max()
    V_target_tau = V_final * 0.632
    
    try:
        tau_row = df[df[volt_col] >= V_target_tau].iloc[0]
        tau_exp = tau_row[time_col]
    except IndexError:
        print("Error: No se pudo encontrar el valor de tau. Revisa tus datos.")
        return

    print(f"Voltaje Final (V_final) detectado: {V_final:.4f} V")
    print(f"Voltaje Objetivo (63.2%): {V_target_tau:.4f} V")
    print(f"-> τ Experimental (tiempo para alcanzar objetivo): {tau_exp:.4f} s")

    C_exp = tau_exp / R_valor
    print(f"-> C Experimental (calculada de τ): {C_exp*1e6:.2f} µF")
    
    A = np.array([[-1/tau_exp]])
    B = np.array([[1/tau_exp]])
    C_ss = np.array([[1]])
    D_ss = np.array([[0]])
    
    ss_model = StateSpace(A, B, C_ss, D_ss)
    u_input = V_final * np.ones_like(t_exp)
    t_sim, y_sim, _ = lsim(ss_model, U=u_input, T=t_exp, X0=[0])
    
    print("Simulación teórica completada.")

    mask = y_exp.notna()
    rmse = np.sqrt(mean_squared_error(y_exp[mask], y_sim[mask]))
    print(f"-> RMSE (Error entre modelo y datos): {rmse:.4f} V")

    plt.figure()
    plt.plot(t_exp, y_exp, label='Experimental (Promedio)', linewidth=3, alpha=0.7)
    plt.plot(t_sim, y_sim, label=f'Modelo Simulado (τ={tau_exp:.2f}s)', linestyle='--', color='red', linewidth=2)
    plt.axvline(x=tau_exp, color='green', linestyle=':', label=f'τ = {tau_exp:.2f}s (al 63.2%)')
    plt.axhline(y=V_target_tau, color='green', linestyle=':')
    
    plt.title(f'Validación del Modelo: R = {R_valor}Ω, C ≈ {C_exp*1e6:.0f}µF')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Voltaje del Capacitor (V)')
    plt.legend()
    plt.grid(True)
    
    output_filename = f'validacion_modelo_{R_valor}ohm.png'
    plt.savefig(output_filename)
    print(f"Gráfico guardado como: {output_filename}")

# Experimento 1 (1kΩ)
analizar_y_validar(
    csv_file='tiempoyvoltaje1k7s_conpromedio.csv',
    R_valor=1000,
    time_col='T',
    volt_col='Vp',
    es_indice_t=True 
)

# Experimento 2 (2kΩ)
analizar_y_validar(
    csv_file='datos_exp_2k.csv', 
    R_valor=2000,
    time_col='T',                
    volt_col='Vp',              
    es_indice_t=True             
)

print("\nAnálisis completado. Mostrando gráficas...")
plt.show() 