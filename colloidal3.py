import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial.distance import pdist, squareform
import time


def torus_dist(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''
    dx = min(abs(p1[0] - p2[0]), box_dims[0] - abs(p1[0] - p2[0]))
    dy = min(abs(p1[1] - p2[1]), box_dims[1] - abs(p1[1] - p2[1]))
    return np.sqrt(dx**2 + dy**2)


def init_grid(X):
    '''Set up the initial grid for particle bookkeeping'''
    global counting_grid

    # initialise with independant empty lists
    for i in range(len(counting_grid[0, :])):
        for j in range(len(counting_grid[:, 0])):
            counting_grid[i, j] = []

    # Fill grid with indicies of particles in that region
    for i, point in enumerate(X):
        counting_grid[int(point[1] * grid_marks / box_dims[1]),
                      int(point[0] * grid_marks / box_dims[0])].append(i)


def init_lattice(d):
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
    print('N = {0}, Eta = {1:.2f}'.format(N, (N * np.pi * r**2) / (box_dims[0] * box_dims[1])))
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

        # Add a single blue particle to get sense of movement
        if index == int(N / 2):
            c = 'b'
        else:
            c = 'r'

        # Add circle
        ax.add_artist(plt.Circle(tuple(pair), r, color=c))
        cross_x_r, cross_x_l, cross_y_t, cross_y_b = crosses_boundary(pair)

        # Conditions for crossing boundaries
        if cross_x_r:
            ax.add_artist(plt.Circle((pair[0] - box_dims[0], pair[1]), r, color='r'))
        elif cross_x_l:
            ax.add_artist(plt.Circle((pair[0] + box_dims[0], pair[1]), r, color='r'))
        if cross_y_t:
            ax.add_artist(plt.Circle((pair[0], pair[1] - box_dims[1]), r, color='r'))
        elif cross_y_b:
            ax.add_artist(plt.Circle((pair[0], pair[1] + box_dims[1]), r, color='r'))

        # Number the particles
        if label:
            plt.annotate(index, xy=(pair[0], pair[1]))

    # Matplotlib
    ax.set_aspect(1)
    ax.set_xlim([0, box_dims[0]])
    ax.set_ylim([0, box_dims[1]])

    # Set grid lines
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


def metropolis_step(X):
    '''For a given state vector X, apply MC Metropolis move and update
        bookkeeping associated with counting grid'''
    global accepted, rejected, counting_grid

    # Original point
    index = np.random.randint(0, len(X))
    point_old = X[index].copy()
    counting_index_old = (int(point_old[1] * grid_marks / box_dims[1]),
                          int(point_old[0] * grid_marks / box_dims[0]))  # the index that the particle appears in in the counting grid

    # Propose new move
    point_new = [0, 0]
    point_new[0] = (point_old[0] + step * (2 * np.random.rand() - 1)) % box_dims[0]
    point_new[1] = (point_old[1] + step * (2 * np.random.rand() - 1)) % box_dims[1]
    counting_index_new = (int(point_new[1] * grid_marks / box_dims[1]),
                          int(point_new[0] * grid_marks / box_dims[0]))

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

    l0 = min(box_dims[0], box_dims[1]) / 2
    d0 = min(l0, radii * r)

    hist = np.histogram(dists, bins=bins, range=(0, d0))
    hist[0][0] = 0

    if first_run:
        circle_areas = np.pi * hist[1]**2
        rings = np.array([circle_areas[i + 1] - circle_areas[i]
                          for i in range(len(circle_areas) - 1)]) * N / (box_dims[0] * box_dims[1])
        dist_array = hist[1][:-1] / r

    first_run = False
    g_of_r = hist[0] / rings
    return g_of_r, dist_array


def simulate(radial_dist=False):
    global X0, N, accepted, rejected

    accepted = 0
    rejected = 0
    X0 = init_lattice(d)
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

        r = 0.1
        d = 2.4 * r
        box_dims = (1, 1)
        step = d - 2 * r
        iters = 500000

        grid_marks = int(box_dims[0] / (2 * r))
        counting_grid = np.empty((grid_marks, grid_marks), dtype=list)

        accepted = 0
        rejected = 0

    #Simulation run
    if True:
        first_run = True
        X, radial_dist, x_ticks = simulate(radial_dist=True)


    #Plotting
    if True:

        plot_circles(X0, title='Initial Config', label=label)
        plot_circles(X, title='Final Config', label=label)

        plt.figure()
        plt.plot(x_ticks, radial_dist)
        plt.xlabel('Radial Speration')
        plt.ylabel('Distribution Function')
        plt.title('N = {0:.0f}, Radius = {1:.2f}, Eta = {2:.2f}'.format(
            N, r, (N * np.pi * r**2) / (box_dims[0] * box_dims[1])))

        plt.show()

    N = 2
    X = np.array([[0.5, 0.24], [0.57, 0.7]])

    d = torus_dist(X[0], X[1])
    dx, dy = X[1] - X[0]
    DX = dy - (4 * r**2 - dx**2)**0.5
    xnew = [X[0][0], X[0][1] + DX]
    X = list(X)
    X.append(xnew)

    plot_circles(X)
    plt.show()
