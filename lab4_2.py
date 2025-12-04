import numpy as np
import matplotlib.pyplot as plt


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
    x = np.zeros(2)
    pts = []

    for _ in range(n_points):
        k = np.random.randint(3)
        x = mats[k] @ x + vecs[k]
        pts.append(x.copy())

    return np.array(pts)


def box_counting_dimension(points, scales=10):
    xs, ys = points[:,0], points[:,1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    sizes = []
    counts = []

    for k in range(1, scales+1):
        eps = (max_x - min_x) / (2**k)
        sizes.append(eps)

        boxes = set()
        for x, y in points:
            i = int((x - min_x) // eps)
            j = int((y - min_y) // eps)
            boxes.add((i,j))
        counts.append(len(boxes))

    log_sizes = np.log(1/np.array(sizes))
    log_counts = np.log(np.array(counts))

    # линейная регрессия
    D = np.polyfit(log_sizes, log_counts, 1)[0]
    return D, sizes, counts


pts = generate_ifs(200000)

D, sizes, counts = box_counting_dimension(pts, scales=12)
print("Оценённая метрическая размерность:", D)


plt.figure(figsize=(7,7))
plt.scatter(pts[:,0], pts[:,1], s=0.2)
plt.title("Аттрактор IFS")
plt.axis('equal')
plt.show()

# График бокса
plt.figure(figsize=(6,4))
plt.plot(np.log(1/np.array(sizes)), np.log(counts), '-o')
plt.title("Box-counting график")
plt.xlabel("log(1/ε)")
plt.ylabel("log N(ε)")
plt.show()
