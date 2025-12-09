import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# ОБЪЯСНЕНИЕ МЕТРИЧЕСКОЙ (ФРАКТАЛЬНОЙ) РАЗМЕРНОСТИ
# =====================================================================
"""
Метрическая размерность (размерность Минковского или box-counting dimension)
показывает, насколько "плотно" фрактал заполняет пространство.

АЛГОРИТМ BOX-COUNTING:
1. Покрываем множество точек сеткой квадратов со стороной ε
2. Считаем N(ε) - количество квадратов, содержащих хотя бы одну точку
3. Уменьшаем ε и повторяем
4. Строим график log(N(ε)) от log(1/ε)
5. Наклон этой прямой и есть метрическая размерность D

ФОРМУЛА:
    D = lim(ε→0) [log(N(ε)) / log(1/ε)]

На практике мы аппроксимируем этот предел линейной регрессией.

ИНТЕРПРЕТАЦИЯ:
- D = 1 → кривая (линия)
- D = 2 → полностью заполненная плоскость
- 1 < D < 2 → фрактал (дробная размерность)

Например, береговая линия имеет D ≈ 1.2-1.3, снежинка Коха D ≈ 1.26
"""
# =====================================================================

A1 = np.array([[-0.829, 0.000],
               [-1.012, 0.092]])
b1 = np.array([ 9.308, -39.740])

A2 = np.array([[0.254, -0.364],
               [0.848, 0.036]])
b2 = np.array([37.376, 67.623])

A3 = np.array([[0.240, 0.532],
               [-0.704,-0.196]])
b3 = np.array([-47.690, -5.668])

mats = [A1, A2, A3]
vecs = [b1, b2, b3]


def generate_ifs(n_points=200000):
    """Генерация точек аттрактора методом случайных итераций (IFS)"""
    x = np.zeros(2)
    pts = []

    for _ in range(n_points):
        k = np.random.randint(3)  # случайный выбор одного из 3 преобразований
        x = mats[k] @ x + vecs[k]  # применяем аффинное преобразование
        pts.append(x.copy())

    return np.array(pts)


def box_counting_dimension(points, scales=10):
    """
    Вычисление метрической размерности методом подсчёта квадратов (box-counting)
    
    Шаги:
    1. Определяем границы множества точек
    2. Для каждого масштаба ε:
       - Делим пространство на квадраты размера ε
       - Считаем, сколько квадратов содержат точки
    3. Находим наклон графика log(N) vs log(1/ε)
    """
    xs, ys = points[:,0], points[:,1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    sizes = []
    counts = []

    for k in range(1, scales+1):
        eps = (max_x - min_x) / (2**k)  # размер квадрата уменьшается в 2 раза
        sizes.append(eps)

        boxes = set()
        for x, y in points:
            i = int((x - min_x) // eps)  # индекс квадрата по X
            j = int((y - min_y) // eps)  # индекс квадрата по Y
            boxes.add((i,j))
        counts.append(len(boxes))  # N(ε) - число занятых квадратов

    log_sizes = np.log(1/np.array(sizes))
    log_counts = np.log(np.array(counts))

    # Линейная регрессия: D = наклон прямой log(N) vs log(1/ε)
    D = np.polyfit(log_sizes, log_counts, 1)[0]
    return D, sizes, counts


# =====================================================================
# ГЕНЕРАЦИЯ 5 АТТРАКТОРОВ С РАЗНЫМ КОЛИЧЕСТВОМ ТОЧЕК
# =====================================================================

n_points_list = [1000, 5000, 20000, 100000, 500000]

print("=" * 60)
print("ВЫЧИСЛЕНИЕ МЕТРИЧЕСКОЙ РАЗМЕРНОСТИ ДЛЯ РАЗНОГО ЧИСЛА ТОЧЕК")
print("=" * 60)
print()

# Фигура 1: 5 аттракторов
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
axes1 = axes1.flatten()

# Фигура 2: 5 графиков метрической размерности (box-counting)
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
axes2 = axes2.flatten()

dimensions = []
all_sizes = []
all_counts = []

for idx, n_points in enumerate(n_points_list):
    print(f"[{idx+1}] Количество точек: {n_points:,}")
    
    # Генерируем аттрактор
    pts = generate_ifs(n_points)
    
    # Вычисляем метрическую размерность
    D, sizes, counts = box_counting_dimension(pts, scales=12)
    dimensions.append(D)
    all_sizes.append(sizes)
    all_counts.append(counts)
    
    print(f"    Метрическая размерность D = {D:.4f}")
    print(f"    Число масштабов: {len(sizes)}")
    print(f"    Диапазон ε: от {sizes[0]:.2f} до {sizes[-1]:.4f}")
    print()
    
    # Рисуем аттрактор (фигура 1)
    axes1[idx].scatter(pts[:,0], pts[:,1], s=0.1, alpha=0.5)
    axes1[idx].set_title(f"Аттрактор N = {n_points:,}\nD = {D:.4f}", fontsize=12)
    axes1[idx].axis('equal')
    axes1[idx].set_xlabel('x')
    axes1[idx].set_ylabel('y')
    
    # Рисуем график box-counting (фигура 2)
    log_inv_eps = np.log(1/np.array(sizes))
    log_N = np.log(np.array(counts))
    
    # Линия регрессии
    coeffs = np.polyfit(log_inv_eps, log_N, 1)
    regression_line = np.poly1d(coeffs)
    
    axes2[idx].plot(log_inv_eps, log_N, 'bo', markersize=6, label='Данные')
    axes2[idx].plot(log_inv_eps, regression_line(log_inv_eps), 'r-', linewidth=2, 
                    label=f'D = {D:.4f}')
    axes2[idx].set_title(f"Box-counting (N = {n_points:,})\nD = {D:.4f}", fontsize=12)
    axes2[idx].set_xlabel('log(1/ε)')
    axes2[idx].set_ylabel('log N(ε)')
    axes2[idx].legend(loc='lower right')
    axes2[idx].grid(True, alpha=0.3)

# Последний subplot фигуры 1 - график сходимости размерности
axes1[5].plot(n_points_list, dimensions, 'bo-', linewidth=2, markersize=8)
axes1[5].set_xscale('log')
axes1[5].set_xlabel('Количество точек (log scale)')
axes1[5].set_ylabel('Метрическая размерность D')
axes1[5].set_title('Сходимость размерности')
axes1[5].grid(True, alpha=0.3)

# Последний subplot фигуры 2 - все графики box-counting на одном
colors = ['blue', 'green', 'red', 'purple', 'orange']
for idx, n_points in enumerate(n_points_list):
    log_inv_eps = np.log(1/np.array(all_sizes[idx]))
    log_N = np.log(np.array(all_counts[idx]))
    axes2[5].plot(log_inv_eps, log_N, 'o-', color=colors[idx], 
                  label=f'N={n_points:,}, D={dimensions[idx]:.3f}', markersize=4)
axes2[5].set_xlabel('log(1/ε)')
axes2[5].set_ylabel('log N(ε)')
axes2[5].set_title('Сравнение Box-counting для всех N')
axes2[5].legend(loc='lower right', fontsize=9)
axes2[5].grid(True, alpha=0.3)

fig1.suptitle('5 АТТРАКТОРОВ С РАЗНЫМ ЧИСЛОМ ТОЧЕК', fontsize=14, fontweight='bold')
fig1.tight_layout()
fig1.savefig('attractors_comparison.png', dpi=150)

fig2.suptitle('ГРАФИКИ МЕТРИЧЕСКОЙ РАЗМЕРНОСТИ (BOX-COUNTING)', fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig('box_counting_all.png', dpi=150)

plt.show()

# =====================================================================
# ДЕТАЛЬНЫЙ ГРАФИК BOX-COUNTING ДЛЯ МАКСИМАЛЬНОГО ЧИСЛА ТОЧЕК
# =====================================================================

print("=" * 60)
print("ДЕТАЛЬНЫЙ АНАЛИЗ BOX-COUNTING")
print("=" * 60)

pts_final = generate_ifs(500000)
D_final, sizes_final, counts_final = box_counting_dimension(pts_final, scales=12)

log_inv_eps = np.log(1/np.array(sizes_final))
log_N = np.log(np.array(counts_final))

# Коэффициенты регрессии
coeffs = np.polyfit(log_inv_eps, log_N, 1)
regression_line = np.poly1d(coeffs)

print(f"Итоговая метрическая размерность: D = {D_final:.4f}")
print()
print("Таблица значений:")
print("-" * 50)
print(f"{'ε':>12} | {'N(ε)':>10} | {'log(1/ε)':>10} | {'log N(ε)':>10}")
print("-" * 50)
for i in range(len(sizes_final)):
    print(f"{sizes_final[i]:>12.4f} | {counts_final[i]:>10} | {log_inv_eps[i]:>10.4f} | {log_N[i]:>10.4f}")
print("-" * 50)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(log_inv_eps, log_N, 'bo', markersize=8, label='Данные')
plt.plot(log_inv_eps, regression_line(log_inv_eps), 'r-', linewidth=2, 
         label=f'Регрессия: D = {D_final:.4f}')
plt.xlabel('log(1/ε)', fontsize=12)
plt.ylabel('log N(ε)', fontsize=12)
plt.title('Box-counting: определение размерности', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(pts_final[:,0], pts_final[:,1], s=0.05, alpha=0.3)
plt.title(f'Аттрактор IFS (N=500,000)\nМетрическая размерность D = {D_final:.4f}', fontsize=14)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig('box_counting_analysis.png', dpi=150)
plt.show()

print()
print("=" * 60)
print("ВЫВОД")
print("=" * 60)
print(f"Метрическая размерность аттрактора ≈ {np.mean(dimensions):.4f}")
print("Это означает, что аттрактор занимает пространство 'между'")
print("одномерной кривой (D=1) и двумерной плоскостью (D=2).")
