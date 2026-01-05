import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 1. ЗАВАНТАЖЕННЯ ТА НАЛАШТУВАННЯ ЧАСУ ---
file_path = r'C:\Users\LOQ\Downloads\Telegram Desktop\A14.txt' 

try:
    df = pd.read_csv(file_path, header=None)
except FileNotFoundError:
    # Генерація тестових даних, якщо файлу немає
    df = pd.DataFrame(np.random.randn(5000, 12))

# Параметри з умови
N = df.shape[0]        # 5000 точок
fs = 500               # 500 Гц
T = 10                 # 10 секунд
dt = 1/fs              # 0.002 с

# Створення вектору часу
time = np.linspace(0, T, N, endpoint=False)

print(f"Дані завантажено. Розмір: {df.shape}")
print(f"Час запису: {T} с, Крок: {dt} с")

# --- 2. ФУНКЦІЯ ДЛЯ СТАТИСТИКИ (включаючи Джині та середні) ---
def gini_mean_difference(series):
    """Обчислення середньої різниці Джині"""
    x = series.values
    n = len(x)
    # Ефективний метод через сортування
    x_sorted = np.sort(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum((2 * index - n - 1) * x_sorted)) / (n * (n - 1))

def get_detailed_stats(df):
    stats_dict = {}
    for col in df.columns:
        data = df[col]
        
        # Перевірка на нормальність (Test D'Agostino's K^2)
        # H0: розподіл нормальний. Якщо p < 0.05, відхиляємо H0.
        k2, p_value = stats.normaltest(data)
        is_normal = "Так" if p_value > 0.05 else "Ні"

        # Середнє геометричне/гармонічне існують тільки для позитивних чисел.
        # Для ЕКГ (де є від'ємні значення) це математично некоректно без зсуву.
        # Рахуємо для абсолютних значень або ставимо NaN, якщо є мінус.
        try:
            h_mean = stats.hmean(data[data > 0]) # Тільки для > 0
            g_mean = stats.gmean(data[data > 0]) 
        except ValueError:
            h_mean = np.nan
            g_mean = np.nan

        stats_dict[f'Channel {col+1}'] = {
            'Mean (Середнє)': np.mean(data),
            'Harmonic Mean (>0)': h_mean,
            'Geometric Mean (>0)': g_mean,
            'Variance (Дисперсія)': np.var(data),
            'Gini Mean Diff': gini_mean_difference(data),
            'Mode': data.mode()[0], # Мода
            'Median': np.median(data),
            'Skewness (Асиметрія)': stats.skew(data),
            'Kurtosis (Ексцес)': stats.kurtosis(data),
            'Normality (p-val)': round(p_value, 4),
            'Is Normal?': is_normal
        }
    return pd.DataFrame(stats_dict).T

# Розрахунок статистики
statistics_df = get_detailed_stats(df)
print("\n--- Основні статистичні параметри ---")
print(statistics_df)

# Збереження статистики в CSV (для звіту)
statistics_df.to_csv('ecg_statistics.csv')

# --- 3. НОРМАЛІЗАЦІЯ ДАНИХ ---
# (X - mean) / std -> Мат. сподівання = 0, Дисперсія = 1
df_normalized = (df - df.mean()) / df.std()

print("\nПеревірка нормалізації (Кан. 1):")
print(f"Mean: {df_normalized[0].mean():.2f}, Std: {df_normalized[0].std():.2f}")

# --- 4. ПОБУДОВА ГРАФІКІВ (Всі 12 каналів) ---
# Побудуємо 4 фігури по 3 канали на кожній для кращої видимості
channels = df.columns
for i in range(0, 12, 3):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Кардіограма та Гістограми (Канали {i+1}-{i+3})')
    
    for j in range(3):
        idx = i + j
        if idx >= 12: break
        
        # Графік сигналу
        axes[j, 0].plot(time, df[idx], color='blue', linewidth=0.8)
        axes[j, 0].set_title(f'Канал {idx+1} (Часова область)')
        axes[j, 0].set_xlabel('Час (сек)')
        axes[j, 0].set_ylabel('Амплітуда')
        axes[j, 0].grid(True)
        
        # Гістограма
        sns.histplot(df[idx], kde=True, ax=axes[j, 1], color='green')
        axes[j, 1].set_title(f'Канал {idx+1} (Гістограма розподілу)')
        axes[j, 1].set_xlabel('Значення')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 5. ГРАФІК НОРМАЛІЗОВАНИХ ДАНИХ (Приклад для 1 каналу) ---
plt.figure(figsize=(12, 4))
plt.plot(time, df_normalized[0], color='red', label='Нормалізований сигнал')
plt.title("Нормалізований сигнал 1-го каналу (Mean=0, Std=1)")
plt.xlabel("Час (сек)")
plt.grid(True)
plt.legend()
plt.show()