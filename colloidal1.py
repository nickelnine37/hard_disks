import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt

# N = 300
r = 0.02
box_dims = (1, 1)
step = r / 2
iters = 5000


# rho_0 = np.pi / (2 * np.sqrt(3))
# rho = (N * np.pi * r**2) / (box_dims[0] * box_dims[1])
# tau = rho_0 / rho

accepted = 0
rejected = 0

bins = 200


def torus_dist(p1, p2):
    x_dist = min(abs(p1[0] - p2[0]), box_dims[0] - abs(p1[0] - p2[0]))
    y_dist = min(abs(p1[1] - p2[1]), box_dims[1] - abs(p1[1] - p2[1]))
    return np.sqrt(x_dist**2 + y_dist**2)


def init_lattice(d, lat_type='hex'):
    global N
    xs = [0]
    ys = [0]
    x = xs[0]
    y = ys[0]
    while True:
        x += d
        if x < box_dims[0] - d:
            xs.append(x)
        else:
            break
    while True:
        y += d * np.sqrt(3) / 2
        if y < box_dims[1] - d * np.sqrt(3) / 2:
            ys.append(y)
        else:
            break
    nx = len(xs)
    ny = len(ys)
    N = nx * ny
    xs = xs * ny
    xs = np.array(xs) + np.array([(int(i / nx) % 2) * (d / 2) for i in range(N)])
    ys2 = []
    for y in ys:
        ys2 += [y] * nx
    X = np.zeros((N, 2))
    X[:, 0] = xs
    X[:, 1] = ys2
    print('N= ', N)
    return X


def init_random(n):
    global N
    X = [[box_dims[0] * np.random.rand(), box_dims[1] * np.random.rand()]]
    i = 0
    loops = 0
    max_loops = n * 10
    while (i < n - 1 and loops < max_loops):
        test_point = [box_dims[0] * np.random.rand(), box_dims[1] * np.random.rand()]
        overlap = False
        for point in X:
            if torus_dist(test_point, point) < 2 * r:
                overlap = True
                break
        if not overlap:
            X.append(test_point)
            i += 1
        loops += 1
    if loops == max_loops:
        print('Failed to add all particles')
    N = i
    print(N, 'added')
    return np.array(X)


def crosses_boundary(position):
    x = position[0]
    y = position[1]
    cross_x_r = x + r > box_dims[0]
    cross_x_l = x - r < 0
    cross_y_t = y + r > box_dims[1]
    cross_y_b = y - r < 0
    return cross_x_r, cross_x_l, cross_y_t, cross_y_b


def plot_circles(X, title=''):
    fig, ax = plt.subplots()
    for index, pair in enumerate(X):
        if index == 200:
            c = 'b'
        else:
            c = 'r'
        ax.add_artist(plt.Circle(tuple(pair), r, color=c))
        cross_x_r, cross_x_l, cross_y_t, cross_y_b = crosses_boundary(pair)
        if cross_x_r:
            ax.add_artist(plt.Circle((pair[0] - box_dims[0], pair[1]), r, color='r'))
        elif cross_x_l:
            ax.add_artist(plt.Circle((pair[0] + box_dims[0], pair[1]), r, color='r'))
        if cross_y_t:
            ax.add_artist(plt.Circle((pair[0], pair[1] - box_dims[1]), r, color='r'))
        elif cross_y_b:
            ax.add_artist(plt.Circle((pair[0], pair[1] + box_dims[1]), r, color='r'))

    ax.set_aspect(1)
    ax.set_xlim([0, box_dims[0]])
    ax.set_ylim([0, box_dims[1]])
    plt.title(title)


def advance_state(X):
    global accepted, rejected
    index = np.random.randint(0, len(X))
    point_old = X[index].copy()
    point_new = [0, 0]
    point_new[0] = (point_old[0] + step * (2 * np.random.rand() - 1)) % box_dims[0]
    point_new[1] = (point_old[1] + step * (2 * np.random.rand() - 1)) % box_dims[1]
    overlap = False
    for i, point in enumerate(X):
        if torus_dist(point_new, point) < 2 * r and i != index:
            rejected += 1
            return X
    accepted += 1
    X[index] = point_new
    return X


def g(X, index, bins, radii):
    global N
    reference_point = X[index]
    dists = []
    l0 = min(box_dims[0], box_dims[1]) / 2

    for i, point in enumerate(X):
        d = torus_dist(point, reference_point)
        if d < min(l0, radii * r):
            dists.append(d)

    hist = np.histogram(dists, bins=bins, range=(0, min(l0, radii * r)))
    hist[0][0] = 0
    circle_areas = np.pi * hist[1]**2
    ring_areas = np.array([circle_areas[i + 1] - circle_areas[i]
                           for i in range(len(circle_areas) - 1)])
    dist_array = hist[1][:-1] / r
    rho = N / (box_dims[0] * box_dims[1])
    g_of_r = hist[0] / (ring_areas * rho)
    return g_of_r, dist_array


if __name__ == '__main__':
    X0 = init_lattice(2.5 * r)
    # X0 = init_random(400)
    X = X0.copy()
    i = 0
    gs = []
    while i < iters:
        X = advance_state(X)
        i += 1
    #     if i > 5000 and i % 1000 == 0:
    #         for j in range(N):
    #             index = np.random.randint(0, N)
    #             gr, dist = g(X, index=index, bins=200, radii=10)
    #             gs.append(gr)
    # gr = np.mean(gs, axis=0)
    # plt.plot(dist, gr)
    plot_circles(X0, 'Initial Config')
    plot_circles(X, 'Final Config')
    print('Acceptance Rate: ', int(((accepted / (accepted + rejected)) * 100)), '%')

    plt.show()
