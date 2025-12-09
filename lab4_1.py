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
    
    # Радиусы для анализа
    radii = np.linspace(5, max_dist * 0.9, 30)
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
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График 1: Агрегат
ax1 = axes[0]
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

# График 2: Зависимость N(r) для расчета размерности
ax2 = axes[1]
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
