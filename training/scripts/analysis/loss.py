import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# 1. Загрузка данных
df = pd.read_csv('training_metrics.csv')
x_data = df['step'].values.astype(float)
y_data = df['loss'].values.astype(float)

# 2. Модель: Экспонента + Степенная + Рациональная + Линейная
def ultimate_func(x, a_exp, b_exp, c_pow, d_pow, r_a, r_b, r_c, m, e):
    x = np.array(x, dtype=float)
    x_s = np.maximum(x, 1e-6)
    return (a_exp * np.exp(-b_exp * x) +
            c_pow * np.power(x_s, -d_pow) +
            (r_a * x + r_b) / (x + r_c) +
            m * x + e)

# Параметры и границы
p0 = [5.0, 1e-2, 10.0, 0.2, -5.0, 500, 200, -1e-6, 3.0]
bounds = ([0, 1e-5, 0, 0, -50, 0, 1, -0.1, 0], [20, 1, 100, 2, 50, 5000, 5000, 0.1, 10])

try:
    # 3. Расчет
    popt, _ = curve_fit(ultimate_func, x_data, y_data, p0=p0, bounds=bounds, maxfev=500000)
    
    target_step = 50000
    x_range = np.linspace(x_data.min(), target_step, 2000)
    y_fit = ultimate_func(x_range, *popt)
    
    # Расчет доверительного интервала (упрощенный статистический подход)
    residuals = y_data - ultimate_func(x_data, *popt)
    std_error = np.std(residuals)
    ci = 1.96 * std_error # 95% интервал на основе отклонения данных

    # 4. Простая и чистая визуализация
    plt.figure(figsize=(12, 7))
    plt.gca().set_facecolor('white')
    
    # Данные (светло-голубые точки)
    plt.scatter(x_data, y_data, s=15, color='#3498db', alpha=0.4, label='Реальный Loss', edgecolors='none')
    
    # Доверительный интервал (мягкая серая заливка)
    plt.fill_between(x_range, y_fit - ci, y_fit + ci, color='gray', alpha=0.15, label='Доверительный интервал')
    
    # Линия аппроксимации (спокойный красный)
    plt.plot(x_range, y_fit, color='#e74c3c', linewidth=2.5, label='Гибридная аппроксимация')
    
    # Точка прогноза
    pred_50k = ultimate_func(target_step, *popt)
    plt.plot(target_step, pred_50k, 'ko', markersize=8)
    plt.annotate(f'Прогноз: {pred_50k:.4f}',
                 xy=(target_step, pred_50k),
                 xytext=(target_step - 12000, pred_50k + 1),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=11)

    # Настройка осей и сетки
    plt.title('Аппроксимация Loss и экстраполяция тренда', fontsize=14)
    plt.xlabel('Шаги обучения (Step)', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, target_step + 1000)
    plt.ylim(y_data.min() - 0.5, y_data.max() + 0.5)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    plt.show()

    print(f"Прогноз на 50 000 шагов: {pred_50k:.6f}")

except Exception as e:
    print(f"Ошибка: {e}")
