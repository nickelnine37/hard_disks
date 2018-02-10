import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time


r = 0.02
box_dims = (1, 1)
step = 3 * r
iters = 50000

accepted = 0
rejected = 0

bins = 200


# Initialise the counting grid
grid_marks = 10
counting_grid = np.empty((grid_marks, grid_marks), dtype=list)

assert box_dims[0] / grid_marks >= 2 * r
assert box_dims[1] / grid_marks >= 2 * r


def torus_dist(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''
    x_dist = min(abs(p1[0] - p2[0]), box_dims[0] - abs(p1[0] - p2[0]))
    y_dist = min(abs(p1[1] - p2[1]), box_dims[1] - abs(p1[1] - p2[1]))
    return np.sqrt(x_dist**2 + y_dist**2)


def init_grid(X):
    '''Set up the initial grid for particle bookkeeping'''
    global counting_grid
    for i in range(len(counting_grid[0, :])):
        for j in range(len(counting_grid[:, 0])):
            counting_grid[i, j] = []
    for i, point in enumerate(X):
        counting_grid[int(point[1] * grid_marks / box_dims[1]),
                      int(point[0] * grid_marks / box_dims[0])].append(i)


def init_lattice(d, lat_type='hex'):
    '''returns vector containing (x,y) coordinates of N particles with
        intermolecular spacing d in a hexagonal lattice'''
    global N
    xs = np.arange(0, box_dims[0] - 2 * r, step=d)
    ys = np.arange(0, box_dims[1] - 2 * r, step=(d * np.sqrt(3) / 2))
    N = len(xs) * len(ys)
    xs, ys = np.meshgrid(xs, ys)
    for i in range(len(ys)):
        if i % 2:
            xs[i] += d / 2
    X = np.array(list(zip(np.matrix.flatten(xs), np.matrix.flatten(ys))))
    print('N =', N)
    return X


def init_random(n):
    '''Disperese n particles randomly over box'''
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
    N = i + 1
    print(N, 'added')
    return np.array(X)


def plot_circles(X, title='', label=False):
    '''Plot particle positions in box'''

    def crosses_boundary(p):
        '''Determine whether particle crosses the boundaries for neat plotting'''
        cross_x_r = p[0] + r > box_dims[0]  # cross x on right?
        cross_x_l = p[0] - r < 0  # cross x on left?
        cross_y_t = p[1] + r > box_dims[1]  # cross y on top?
        cross_y_b = p[1] - r < 0  # cross y on bottom?
        return cross_x_r, cross_x_l, cross_y_t, cross_y_b

    fig, ax = plt.subplots()

    for index, pair in enumerate(X):
        if index == int(N / 2):
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

        if label:
            plt.annotate(index, xy=(pair[0], pair[1]))

    ax.set_aspect(1)
    ax.set_xlim([0, box_dims[0]])
    ax.set_ylim([0, box_dims[1]])

    x_spacing = box_dims[0] / grid_marks  # x grid spacing.
    y_spacing = box_dims[1] / grid_marks  # y grid spacing.
    minorLocator_x = MultipleLocator(x_spacing)
    minorLocator_y = MultipleLocator(y_spacing)

    ax.yaxis.set_minor_locator(minorLocator_y)
    ax.xaxis.set_minor_locator(minorLocator_x)
    ax.grid(which='minor')

    plt.title(title)


def extract_3x3(A, index):
    '''For a given index in an array, return the 3x3 matrix that surrounds
        that index, wrapping with torus topology at edges'''
    M3x3 = A.take(range(index[0] - 1, index[0] + 2), axis=0, mode='wrap')\
            .take(range(index[1] - 1, index[1] + 2), axis=1, mode='wrap')
    return M3x3


def advance_state_old(X):
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


def advance_state_new(X):
    '''For a given state vector X, apply MC Metropolis move and update
        bookkeeping associated with counting grid'''
    global accepted, rejected, counting_grid

    index = np.random.randint(0, len(X))
    point_old = X[index].copy()

    counting_index_old = (int(point_old[1] * grid_marks / box_dims[1]),
                          int(point_old[0] * grid_marks / box_dims[0]))  # the index that the particle appears in in the counting grid

    point_new = [0, 0]
    point_new[0] = (point_old[0] + step * (2 * np.random.rand() - 1)) % box_dims[0]
    point_new[1] = (point_old[1] + step * (2 * np.random.rand() - 1)) % box_dims[1]

    counting_index_new = (int(point_new[1] * grid_marks / box_dims[1]),
                          int(point_new[0] * grid_marks / box_dims[0]))

    # Use counting grid to check for near neighbours
    neighbourhood_indexes = [neighbour for neighbours in extract_3x3(
        counting_grid, counting_index_new).flatten() for neighbour in neighbours]

    overlap = False
    for point_index in neighbourhood_indexes:
        if torus_dist(point_new, X[point_index]) < 2 * r and point_index != index:
            rejected += 1
            return X

    accepted += 1
    X[index] = point_new

    if counting_index_old != counting_index_new:
        counting_grid[counting_index_old].remove(index)
        counting_grid[counting_index_new].append(index)

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

    label = True

    Ns = [20, 60, 100, 140, 180, 220, 260, 300, 340]

    r = 0.02
    box_dims = (1, 1)
    step = 5 * r
    iters = 50000

    grid_marks = int(box_dims[0] / (2 * r))
    counting_grid = np.empty((grid_marks, grid_marks), dtype=list)

    old_times = []
    new_times = []

    for n in Ns:

        print(n)

        # naive check
        accepted = 0
        rejected = 0
        X0 = init_random(n)
        X = X0.copy()
        j = 0
        t0 = time.time()
        while j < iters:
            X = advance_state_old(X)
            j += 1
        old_times.append(time.time() - t0)
        print('Acceptance Rate: ', int(((accepted / (accepted + rejected)) * 100)), '%')

        # bookkeeping
        accepted = 0
        rejected = 0
        X0 = init_random(n)
        X = X0.copy()
        init_grid(X)
        j = 0
        t0 = time.time()
        while j < iters:
            X = advance_state_new(X)
            j += 1
        new_times.append(time.time() - t0)
        print('Acceptance Rate: ', int(((accepted / (accepted + rejected)) * 100)), '%')

        print()

    plt.scatter(Ns, old_times, label='Checking Every Particle')
    plt.scatter(Ns, new_times, label='Grid Bookkeeping')
    plt.xlabel('Number of Particles')
    plt.ylabel('Time s')
    plt.legend()
    plt.show()

    # plot_circles(X0, title='Initial Config old', label=label)
    # plot_circles(X, title='Final Config old', label=label)
    # gs = []
    # if i > 5000 and i % 1000 == 0:
    #     for j in range(int(N / 2)):
    #         index = np.random.randint(0, N)
    #         gr, dist = g(X, index=index, bins=200, radii=10)
    #         gs.append(gr)
    # gr = np.mean(gs, axis=0)
    # plt.plot(dist, gr)
