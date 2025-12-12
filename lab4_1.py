import numpy as np
import matplotlib.pyplot as plt

# Seed для воспроизводимости (меняй для разных результатов)
SEED = 42
np.random.seed(SEED)

# Параметры модели
N_PARTICLES = 3000  # количество частиц
MAX_STEPS = 50000   # максимум шагов для одной частицы

GRID_SIZE = 500
CENTER = GRID_SIZE // 2

# Начальный радиус окружности запуска частиц
START_RADIUS = 10
# Радиус будет увеличиваться по мере роста агрегата


# --------------------------
# ФУНКЦИИ
# --------------------------
def random_step():
    """Случайный шаг в одном из 4 направлений с равной вероятностью."""
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    return dirs[np.random.randint(4)]


def spawn_on_circle(radius):
    """Создать частицу на окружности заданного радиуса."""
    angle = np.random.uniform(0, 2 * np.pi)
    x = int(CENTER + radius * np.cos(angle))
    y = int(CENTER + radius * np.sin(angle))
    return x, y


def is_sticking(x, y, grid):
    """Проверяем соседей по сетке (прилипание к центру или к другим точкам)."""
    return (
        grid[y + 1, x] or
        grid[y - 1, x] or
        grid[y, x + 1] or
        grid[y, x - 1]
    )


def calculate_fractal_dimension(points, center):
    """
    Расчет фрактальной размерности методом подсчета частиц в кругах.
    N(r) ~ r^D => log(N) = D * log(r) + const
    """
    if len(points) < 10:
        return None, None, None
    
    # Расстояния от центра до всех точек
    distances = np.sqrt((points[:, 0] - center) ** 2 + (points[:, 1] - center) ** 2)
    max_dist = np.max(distances)
    
    # Радиусы для анализа - логарифмическая сетка для равномерного покрытия в лог-координатах
    # Это дает более точную оценку размерности, так как точки равномерно распределены в log(r)
    min_r = max(5, max_dist * 0.05)  # Минимальный радиус (не менее 5, но не менее 5% от максимума)
    max_r = max_dist * 0.9  # Максимальный радиус (90% от максимума, чтобы избежать краевых эффектов)
    radii = np.logspace(np.log10(min_r), np.log10(max_r), 30)  # 30 радиусов в логарифмическом масштабе
    counts = []
    
    for r in radii:
        count = np.sum(distances <= r)
        if count > 0:
            counts.append(count)
        else:
            counts.append(1)
    
    counts = np.array(counts)
    
    # Линейная регрессия в лог-координатах
    log_r = np.log(radii)
    log_n = np.log(counts)
    
    # Убираем точки где count не меняется
    valid = np.diff(log_n, prepend=log_n[0]) > 0
    valid[0] = True
    
    if np.sum(valid) < 5:
        return None, None, None
    
    coeffs = np.polyfit(log_r[valid], log_n[valid], 1)
    D = coeffs[0]  # фрактальная размерность
    
    return D, radii, counts


# --------------------------
# МОДЕЛИРОВАНИЕ
# --------------------------
# Решетка
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

# Центральный зародыш
grid[CENTER, CENTER] = 1
stuck_points = [(CENTER, CENTER)]

# Текущий максимальный радиус агрегата
max_aggregate_radius = 1
launch_radius = START_RADIUS

# Список для сохранения снимков агрегата на разных этапах
snapshots = []  # Каждый элемент: (количество_точек, список_точек)

# Определяем этапы для сохранения снимков
# Будем сохранять на 100, 500, 1000, 1500, 2000, 2500, 3000 точках
snapshot_stages = [100, 500, 1000, 1500, 2000, 2500, 3000]

print("Запуск моделирования DLA...")

for i in range(N_PARTICLES):
    # Частица стартует с окружности
    x, y = spawn_on_circle(launch_radius)
    
    for _ in range(MAX_STEPS):
        dx, dy = random_step()
        x += dx
        y += dy
        
        # Расстояние от центра
        dist = np.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
        
        # Если ушла слишком далеко — перезапуск
        if dist > launch_radius * 2:
            x, y = spawn_on_circle(launch_radius)
            continue
        
        # Проверка границ
        if x <= 2 or x >= GRID_SIZE - 3 or y <= 2 or y >= GRID_SIZE - 3:
            x, y = spawn_on_circle(launch_radius)
            continue
        
        # Проверка прилипания
        if is_sticking(x, y, grid):
            grid[y, x] = 1
            stuck_points.append((x, y))
            
            # Сохраняем снимок, если достигли нужного этапа
            current_count = len(stuck_points)
            if current_count in snapshot_stages:
                snapshots.append((current_count, stuck_points.copy()))
            
            # Обновляем радиус агрегата
            particle_dist = np.sqrt((x - CENTER) ** 2 + (y - CENTER) ** 2)
            if particle_dist > max_aggregate_radius:
                max_aggregate_radius = particle_dist
                # Увеличиваем радиус запуска
                launch_radius = max(START_RADIUS, int(max_aggregate_radius + 10))
            break
    
    # Прогресс
    if (i + 1) % 500 == 0:
        print(f"Частиц: {i + 1}/{N_PARTICLES}, радиус агрегата: {max_aggregate_radius:.1f}")

print("Моделирование завершено!")

# Добавляем финальный снимок, если его еще нет
if len(stuck_points) not in snapshot_stages:
    snapshots.append((len(stuck_points), stuck_points.copy()))

# --------------------------
# РАСЧЕТ ФРАКТАЛЬНОЙ РАЗМЕРНОСТИ
# --------------------------
stuck_points = np.array(stuck_points)
D, radii, counts = calculate_fractal_dimension(stuck_points, CENTER)

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ:")
print("=" * 50)
print(f"Количество прилипших частиц: {len(stuck_points)}")
print(f"Максимальный радиус агрегата: {max_aggregate_radius:.2f}")
if D is not None:
    print(f"Фрактальная (метрическая) размерность D = {D:.3f}")
    print(f"(Теоретическое значение для DLA ≈ 1.71)")
print("=" * 50)

# --------------------------
# ВИЗУАЛИЗАЦИЯ
# --------------------------

# График 1: Все этапы роста агрегата
n_snapshots = len(snapshots)
n_cols = 3
n_rows = (n_snapshots + n_cols - 1) // n_cols  # Округление вверх

fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
if n_snapshots == 1:
    axes1 = [axes1]
else:
    axes1 = axes1.flatten()

for idx, (count, points) in enumerate(snapshots):
    ax = axes1[idx]
    points_arr = np.array(points)
    X = points_arr[:, 0] - CENTER
    Y = points_arr[:, 1] - CENTER
    ax.scatter(X, Y, s=1, color="blue", alpha=0.7)
    ax.scatter(0, 0, s=50, color="red", marker="x", label="Центр")
    
    # Расчет размерности для этого этапа
    D_stage, _, _ = calculate_fractal_dimension(points_arr, CENTER)
    
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title = f"N={count}"
    if D_stage is not None:
        title += f", D={D_stage:.2f}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Скрываем лишние подграфики
for idx in range(n_snapshots, len(axes1)):
    axes1[idx].axis('off')

plt.suptitle("Рост DLA агрегата на разных этапах", fontsize=14, y=1.0)
plt.tight_layout()
plt.show()

# График 2: Финальный агрегат и расчет размерности
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# График 2.1: Финальный агрегат
ax1 = axes2[0]
X = stuck_points[:, 0] - CENTER
Y = stuck_points[:, 1] - CENTER
ax1.scatter(X, Y, s=1, color="blue", alpha=0.7)
ax1.scatter(0, 0, s=50, color="red", marker="x", label="Центр (зародыш)")
ax1.set_aspect("equal")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title(f"DLA агрегат (N={len(stuck_points)}, D={D:.2f})" if D else "DLA агрегат")
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2.2: Зависимость N(r) для расчета размерности
ax2 = axes2[1]
if radii is not None:
    ax2.loglog(radii, counts, 'bo-', markersize=4, label='N(r) — данные')
    # Линия регрессии
    if D is not None:
        fit_line = np.exp(np.polyfit(np.log(radii), np.log(counts), 1)[1]) * radii ** D
        ax2.loglog(radii, fit_line, 'r--', linewidth=2, label=f'Аппроксимация: D={D:.3f}')
    ax2.set_xlabel("Радиус r")
    ax2.set_ylabel("Число частиц N(r)")
    ax2.set_title("Расчет метрической размерности\n$N(r) \\sim r^D$")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()
