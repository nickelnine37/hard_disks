import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial.distance import pdist, squareform
import time


def torus_dist1(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''
    dx = min(abs(p1[0] - p2[0]), Lx - abs(p1[0] - p2[0]))
    dy = min(abs(p1[1] - p2[1]), Ly - abs(p1[1] - p2[1]))
    return np.sqrt(dx**2 + dy**2)


def torus_dist2(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''

    dx, dy = abs(p1 - p2)

    if dx > Lx2:
        dx = Lx - dx
    if dy > Ly2:
        dy = Ly - dy

    return (dx**2 + dy**2)**0.5


def torus_dist3(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''
    dx, dy = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
    dx, dy = min(dx, Lx - dx), min(dy, Ly - dy)
    return np.sqrt(dx**2 + dy**2)


def torus_dist4(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''

    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])

    if dx > Lx2:
        dx = Lx - dx
    if dy > Ly2:
        dy = Ly - dy

    return (dx**2 + dy**2)**0.5


def init_grid(X):
    '''Set up the initial grid for particle bookkeeping'''
    global counting_grid

    # initialise with independant empty lists
    for i in range(len(counting_grid[0, :])):
        for j in range(len(counting_grid[:, 0])):
            counting_grid[i, j] = []

    # Fill grid with indicies of particles in that region
    for i, point in enumerate(X):
        counting_grid[int(point[1] * grid_marks_y / Ly),
                      int(point[0] * grid_marks_x / Lx)].append(i)


def init_lattice(d):
    '''returns vector containing (x,y) coordinates of N particles with
        intermolecular spacing d in a hexagonal lattice'''
    global N
    xs = np.arange(0, Lx - 2 * r, step=d)
    ys = np.arange(0, Ly - 2 * r, step=(d * np.sqrt(3) / 2))
    N = len(xs) * len(ys)
    xs, ys = np.meshgrid(xs, ys)
    for i in range(len(ys)):
        if i % 2:
            xs[i] += d / 2
    X = np.array(list(zip(np.matrix.flatten(xs), np.matrix.flatten(ys))))
    print('N = {0}, Eta = {1:.2f}'.format(N, (N * np.pi * r**2) / (Lx * Ly)))
    return X


def init_lattice_eta(eta):
    global Lx, Ly, d

    if np.sqrt(N) - int(np.sqrt(N)) != 0:
        raise Exception("N must be a square number")  # Confirm N is a square number

    k = np.sqrt(3) / 2
    Lx = np.sqrt(N * (r**2) * np.pi / (eta * k))
    Ly = np.sqrt(k * N * (r**2) * np.pi / eta)
    d = Lx / np.sqrt(N)

    xs = np.arange(0, Lx - 2 * r, step=d)
    ys = np.arange(0, Ly - 2 * r * k, step=d * k)

    xs, ys = np.meshgrid(xs, ys)
    for i in range(len(ys)):
        if i % 2:
            xs[i] += d / 2
    X = np.array(list(zip(np.matrix.flatten(xs), np.matrix.flatten(ys))))
    print('N = {0}, Eta = {1:.2f}'.format(N, (N * np.pi * r**2) / (Lx * Ly)))
    return X


def init_random(n):
    '''Disperese n particles randomly over box'''
    global N
    X = [[Lx * np.random.rand(), Ly * np.random.rand()]]
    i = 0
    loops = 0
    max_loops = n * 10
    while (i < n - 1 and loops < max_loops):
        test_point = [Lx * np.random.rand(), Ly * np.random.rand()]
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
        cross_x_r = p[0] + r > Lx  # cross x on right?
        cross_x_l = p[0] - r < 0  # cross x on left?
        cross_y_t = p[1] + r > Ly  # cross y on top?
        cross_y_b = p[1] - r < 0  # cross y on bottom?
        return cross_x_r, cross_x_l, cross_y_t, cross_y_b

    fig, ax = plt.subplots()

    for index, pair in enumerate(X):

        # Add a single blue particle to get sense of movement
        if index == int(N / 2) + int(np.sqrt(N) / 2):
            c = 'b'
        else:
            c = 'r'

        # Add circle
        ax.add_artist(plt.Circle(tuple(pair), r, color=c))
        cross_x_r, cross_x_l, cross_y_t, cross_y_b = crosses_boundary(pair)

        # Conditions for crossing boundaries
        if cross_x_r:
            ax.add_artist(plt.Circle((pair[0] - Lx, pair[1]), r, color='r'))
            if cross_y_b:
                ax.add_artist(plt.Circle(
                    (pair[0] - Lx, pair[1] + Ly), r, color='r'))
        elif cross_x_l:
            ax.add_artist(plt.Circle((pair[0] + Lx, pair[1]), r, color='r'))
            if cross_y_b:
                ax.add_artist(plt.Circle(
                    (pair[0] + Lx, pair[1] + Ly), r, color='r'))
        if cross_y_t:
            ax.add_artist(plt.Circle((pair[0], pair[1] - Ly), r, color='r'))
        elif cross_y_b:
            ax.add_artist(plt.Circle((pair[0], pair[1] + Ly), r, color='r'))

        # Number the particles
        if label:
            plt.annotate(index, xy=(pair[0], pair[1]))

    # Matplotlib
    ax.set_aspect(1)
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    # Set grid lines
    x_spacing = Lx / grid_marks_x  # x grid spacing.
    y_spacing = Ly / grid_marks_y  # y grid spacing.
    minorLocator_x = MultipleLocator(x_spacing)
    minorLocator_y = MultipleLocator(y_spacing)

    ax.yaxis.set_minor_locator(minorLocator_y)
    ax.xaxis.set_minor_locator(minorLocator_x)
    ax.grid(which='minor')

    plt.title(title)


def plot_radial_dist(x_ticks, radial_dist):
    fig, ax = plt.subplots()
    plt.plot(x_ticks, radial_dist)
    plt.xlabel('Radial Speration')
    plt.ylabel('Distribution Function')
    title = 'N = {0:.0f}, Eta = {1:.2f}'.format(
        N, (N * np.pi * r**2) / (Lx * Ly))
    ax.set_ylim([0, 6])
    plt.title(title)
    return title


def extract_3x3(A, index):
    '''For a given index in an array, return the 3x3 matrix that surrounds
        that index, wrapping with torus topology at edges'''
    M3x3 = A.take(range(index[0] - 1, index[0] + 2), axis=0, mode='wrap')\
        .take(range(index[1] - 1, index[1] + 2), axis=1, mode='wrap')
    return M3x3


def metropolis_step(X):
    '''For a given state vector X, apply MC Metropolis move and update
        bookkeeping associated with counting grid'''
    global accepted, rejected, counting_grid

    # Original point
    index = np.random.randint(0, len(X))
    point_old = X[index].copy()
    counting_index_old = (int(point_old[1] * grid_marks_y / Ly),
                          int(point_old[0] * grid_marks_x / Lx))  # the index that the particle appears in in the counting grid

    # Propose new move
    point_new = [0, 0]
    point_new[0] = (point_old[0] + step * (2 * np.random.rand() - 1)) % Lx
    point_new[1] = (point_old[1] + step * (2 * np.random.rand() - 1)) % Ly
    counting_index_new = (int(point_new[1] * grid_marks_y / Ly),
                          int(point_new[0] * grid_marks_x / Lx))

    # Use counting grid to check for nearest neighbours
    neighbour_indicies = [neighbour for neighbours in
                          extract_3x3(counting_grid, counting_index_new).flatten()
                          for neighbour in neighbours]

    # Determine whether overlap occurs
    for point_index in neighbour_indicies:
        if torus_dist(point_new, X[point_index]) < 2 * r and point_index != index:
            rejected += 1
            return X

    # Else accept move
    accepted += 1
    X[index] = point_new

    # Determine whether grid counts need to be changed
    if counting_index_old != counting_index_new:
        counting_grid[counting_index_old].remove(index)
        counting_grid[counting_index_new].append(index)

    return X


def g(dists, bins, radii):
    global N, rings, first_run, dist_array

    l0 = min(Lx, Ly) / 2
    d0 = min(l0, radii * r)

    hist = np.histogram(dists, bins=bins, range=(0, d0))
    hist[0][0] = 0

    if first_run:
        circle_areas = np.pi * hist[1]**2
        rings = np.array([circle_areas[i + 1] - circle_areas[i]
                          for i in range(len(circle_areas) - 1)]) * N / (Lx * Ly)
        dist_array = hist[1][:-1] / r

    first_run = False
    g_of_r = hist[0] / rings
    return g_of_r, dist_array


def simulate(X0, eta, radial_dist=False):
    global N, accepted, rejected

    accepted = 0
    rejected = 0
    X = X0.copy()
    init_grid(X)
    j = 0
    t0 = time.time()

    if radial_dist:

        distribution_list = []
        while j < iters:
            X = metropolis_step(X)
            j += 1

            if j > 50000 and j % 1000 == 0:
                dist_array = squareform(pdist(X, torus_dist))
                for dists in dist_array:
                    radial_distribution, ticks = g(dists, bins=200, radii=12)
                    distribution_list.append(radial_distribution)

        mean_distribution = np.mean(distribution_list, axis=0)

        print('Time taken: {0:.2f} s for {1} iterations'.format(time.time() - t0, iters))
        print('Acceptance Rate: {:.0f}%'.format(((accepted / (accepted + rejected)) * 100)))

        return X, mean_distribution, ticks

    else:

        while j < iters:
            X = metropolis_step(X)
            j += 1

        print('Time taken: {0:.2f} s for {1} iterations'.format(time.time() - t0, iters))
        print('Acceptance Rate: {:.0f}%'.format(((accepted / (accepted + rejected)) * 100)))

        return X


if __name__ == '__main__':

    # Variable initialisation`
    if True:
        label = False

        r = 0.02
        N = 256

        iters = 500

        accepted = 0
        rejected = 0

    # Simulation run
    if True:
        X0 = init_lattice_eta(0.7)
        Lx2 = Lx / 2
        Ly2 = Ly / 2
        for func in [torus_dist1, torus_dist2, torus_dist3, torus_dist4]:
            t0 = time.time()
            for i in range(50):
                sf = squareform(pdist(X0, func))
            print('time: ', time.time() - t0)
        etas = np.arange(0.6, 0.75, 0.01)

        # for eta in etas:
        #
        #     first_run = True
        #
        #     step = d - 2 * r
        #     grid_marks_x = int(Lx / (2 * r))
        #     grid_marks_y = int(Ly / (2 * r))
        #     counting_grid = np.empty((grid_marks_y, grid_marks_x), dtype=list)
        #
        # X, mean_distribution, ticks = simulate(X0, eta, radial_dist=True)
        # title = plot_radial_dist(ticks, mean_distribution)
        # plt.savefig(title + '.png')

plt.close('all')
