import numpy as np
import matplotlib.pyplot as plt


N_PARTICLES = 8000
MAX_STEPS = 5000

GRID_SIZE = 500
OFFSET = GRID_SIZE // 2

STICK_Y = 0
STEP = 1

# стартуем не слишком высоко
START_Y = OFFSET + 50
START_X_RANGE = 150

# решетка
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

# линия прилипания
grid[OFFSET + STICK_Y, :] = 1


# --------------------------
# ФУНКЦИИ
# --------------------------
def random_step():
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    return dirs[np.random.randint(4)]

def is_sticking(x, y):
    """Проверяем соседей по сетке."""
    return (
        grid[y+1, x] or
        grid[y-1, x] or
        grid[y, x+1] or
        grid[y, x-1]
    )


stuck_points = []

for _ in range(N_PARTICLES):
    x = np.random.randint(-START_X_RANGE, START_X_RANGE) + OFFSET
    y = START_Y

    for _ in range(MAX_STEPS):
        dx, dy = random_step()
        x += dx
        y += dy

        # вышел за границу — перезапуск
        if x <= 2 or x >= GRID_SIZE-3 or y <= 2 or y >= GRID_SIZE-3:
            x = np.random.randint(-START_X_RANGE, START_X_RANGE) + OFFSET
            y = START_Y
            continue

        if is_sticking(x, y):
            grid[y, x] = 1
            stuck_points.append((x, y))
            break


stuck_points = np.array(stuck_points)
X = stuck_points[:, 0] - OFFSET
Y = stuck_points[:, 1] - OFFSET

plt.figure(figsize=(8, 8))
plt.scatter(X, Y, s=2, color="blue")

plt.axhline(0, color="black", linestyle="--")
plt.gca().set_aspect("equal")

plt.xlim(-80, 80)
plt.ylim(-10, 120)

plt.title("Случайное блуждание с прилипанием")
plt.show()
