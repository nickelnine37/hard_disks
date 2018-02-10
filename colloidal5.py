import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial.distance import pdist, squareform
import time
import fastrand
import matplotlib.cm
import seaborn as sns

pi = np.pi
sqrt = np.sqrt


def torus_dist(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''

    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])

    if dx > Lx2:
        dx = Lx - dx
    if dy > Ly2:
        dy = Ly - dy

    return (dx**2 + dy**2)**0.5


def torus_angle(ref_point, other_point):

    dx = other_point[0] - ref_point[0]
    dy = other_point[1] - ref_point[1]

    if abs(dx) > Lx2:
        if dx < 0:
            dx += Lx
        else:
            dx += -Lx
    if abs(dy) > Ly2:
        if dy < 0:
            dy += Ly
        else:
            dy += -Ly

    phi = np.arctan2(dy, dx)
    if phi < 0:
        phi += 2 * pi

    return phi


def init_grid(X):
    '''Set up the initial grid for particle bookkeeping'''
    global grid_marks_x, grid_marks_y, counting_grid

    # Set optimal grid spacing
    grid_marks_x = int(Lx / (2 * r))
    grid_marks_y = int(Ly / (2 * r))

    # initialise with independant empty lists
    counting_grid = np.empty((grid_marks_y, grid_marks_x), dtype=list)
    for i in range(grid_marks_y):
        for j in range(grid_marks_x):
            counting_grid[i, j] = []

    # Fill grid with indicies of particles in that region
    for i, point in enumerate(X):
        counting_grid[int(point[1] * grid_marks_y / Ly),
                      int(point[0] * grid_marks_x / Lx)].append(i)

    return


def init_lattice_eta(eta):
    '''Returns the lattice configuration of N particles for a given value
        of eta (space filling constant). Globally updates Lx and Ly variables'''
    global Lx, Ly, d, Lx2, Ly2

    # Confirm N is a square number
    if sqrt(N) - int(sqrt(N)) != 0:
        raise Exception("N must be a square number")

    # Compute Lx and Ly variables
    k = sqrt(3) / 2
    Lx = sqrt(N * (r**2) * pi / (eta * k))
    Ly = sqrt(k * N * (r**2) * pi / eta)
    Lx2 = Lx / 2
    Ly2 = Ly / 2

    # Initialise square lattice
    d = Lx / sqrt(N)
    xs = np.arange(0, Lx - 2 * r, step=d)
    ys = np.arange(0, Ly - 2 * r * k, step=d * k)
    xs, ys = np.meshgrid(xs, ys)

    # Hexagonalise
    for i in range(len(ys)):
        if i % 2:
            xs[i] += d / 2

    # Flatten and trasnform into numpy array
    X = np.array(list(zip(np.matrix.flatten(xs), np.matrix.flatten(ys))))

    # Check number of particles and Eta value are as expected
    print('N = {0}, Eta = {1:.3f}'.format(N, (N * pi * r**2) / (Lx * Ly)))
    return X


def init_random(n):
    '''Disperese n particles randomly over box'''
    global N, Lx, Ly
    Lx, Ly = 1, 1
    X = [[np.random.rand(), np.random.rand()]]
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


def plot_circles(X,
                 title='',
                 circles=True,
                 label=False,
                 coloured_index=None,
                 alpha=1,
                 fill=False,
                 LOP=False,
                 LDP=False,
                 resolution=None,
                 gridlines=True):
    '''Plot particle positions in box'''

    # plot Cirlces
    if circles:
        def crosses_boundary(p):
            '''Determine whether particle crosses the boundaries for neat plotting'''
            cross_x_r = p[0] + r > Lx  # cross x on right?
            cross_x_l = p[0] - r < 0  # cross x on left?
            cross_y_t = p[1] + r > Ly  # cross y on top?
            cross_y_b = p[1] - r < 0  # cross y on bottom?
            return cross_x_r, cross_x_l, cross_y_t, cross_y_b

        fig, ax = plt.subplots()
        if coloured_index != None:
            neighbour_indicies, _ = nearest_n_neighbours(
                X[coloured_index], n=6, All=False, particle=True)
        else:
            neighbour_indicies = []

        for index, pair in enumerate(X):

            # Add a single blue particle to get sense of movement
            if index == coloured_index:
                c = 'black'
            elif index in neighbour_indicies:
                c = 'black'
            else:
                c = 'black'

            # Add circle
            ax.add_artist(plt.Circle(tuple(pair), r, color=c, alpha=alpha, fill=fill))
            cross_x_r, cross_x_l, cross_y_t, cross_y_b = crosses_boundary(pair)

            # Conditions for crossing boundaries
            if cross_x_r:
                ax.add_artist(plt.Circle((pair[0] - Lx, pair[1]),
                                         r, color=c, alpha=alpha, fill=fill))
                if cross_y_b:
                    ax.add_artist(plt.Circle(
                        (pair[0] - Lx, pair[1] + Ly), r, color=c, alpha=alpha, fill=fill))
            elif cross_x_l:
                ax.add_artist(plt.Circle((pair[0] + Lx, pair[1]),
                                         r, color=c, alpha=alpha, fill=fill))
                if cross_y_b:
                    ax.add_artist(plt.Circle(
                        (pair[0] + Lx, pair[1] + Ly), r, color='r'))
            if cross_y_t:
                ax.add_artist(plt.Circle((pair[0], pair[1] - Ly),
                                         r, color=c, alpha=alpha, fill=fill))
            elif cross_y_b:
                ax.add_artist(plt.Circle((pair[0], pair[1] + Ly),
                                         r, color=c, alpha=alpha, fill=fill))

            # Number the particles
            if label:
                plt.annotate(index, xy=(pair[0], pair[1]))

        # Matplotlib
        ax.set_aspect(1)
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])

    # Set grid lines
    if gridlines:
        x_spacing = Lx / grid_marks_x  # x grid spacing.
        y_spacing = Ly / grid_marks_y  # y grid spacing.
        minorLocator_x = MultipleLocator(x_spacing)
        minorLocator_y = MultipleLocator(y_spacing)

        ax.yaxis.set_minor_locator(minorLocator_y)
        ax.xaxis.set_minor_locator(minorLocator_x)
        ax.grid(which='minor')

    # Local Orientation Perameter
    if LOP:

        vim = -180
        vmax = 180
        cm = orientation_colormap(resolution=resolution)

        im1 = plt.imshow(cm,
                         extent=(0, Lx, 0, Ly),
                         origin='lower',
                         cmap=matplotlib.cm.hsv,
                         vmin=vim,
                         vmax=vmax)

        plt.colorbar()

    # Local Disorder Peramater
    if LDP:

        # vim = 0
        # vmax = 1
        cm = orientation_colormap(resolution=resolution)

        im1 = plt.imshow(cm,
                         extent=(0, Lx, 0, Ly),
                         origin='lower',
                         cmap=matplotlib.cm.winter,)
        #  vmin=vim,
        #  vmax=vmax)

        plt.colorbar()

    plt.title(title)


def plot_radial_dist(x_ticks, radial_dist, poly_order=5):
    '''Function that scatters all points recorded for the radial distribution
        function with a polynomial fit'''

    xp = np.linspace(x_ticks[0], x_ticks[-1], 5000)

    x_ticks = x_ticks[np.where(x_ticks > 2)]
    radial_dist = radial_dist[np.where(x_ticks > 2)]

    p = np.poly1d(np.polyfit(x_ticks[1:], radial_dist[1:], poly_order))

    fig, ax = plt.subplots()

    plt.scatter(x_ticks, radial_dist, color='r', alpha=0.8)
    plt.plot(xp, p(xp), color='black')

    plt.xlabel('Radial Speration')
    plt.ylabel('Distribution Function')
    title = 'N = {0:.0f}, Eta = {1:.3f}'.format(
        N, (N * pi * r**2) / (Lx * Ly))

    ax.set_ylim([0.95 * min(radial_dist), 1.1 * max(radial_dist)])

    plt.title(title)
    print('Eta: {0:.3f}, g(2r): {1}'.format(eta, p(2)))
    return title


def extract_3x3(A, index):
    '''For a given index in an array, return the 3x3 matrix that surrounds
        that index, wrapping with torus topology at edges'''
    M3x3 = A.take(range(index[0] - 1, index[0] + 2), axis=0, mode='wrap')\
            .take(range(index[1] - 1, index[1] + 2), axis=1, mode='wrap')
    return M3x3


def extract_5x5(A, index):
    '''For a given index in an array, return the 3x3 matrix that surrounds
        that index, wrapping with torus topology at edges'''
    M5x5 = A.take(range(index[0] - 3, index[0] + 4), axis=0, mode='wrap')\
            .take(range(index[1] - 3, index[1] + 4), axis=1, mode='wrap')
    return M5x5


def nearest_n_neighbours(point, n=None, All=False, particle=True):

    counting_index = (int(point[1] * grid_marks_y / Ly),
                      int(point[0] * grid_marks_x / Lx))

    neighbour_indicies = [neighbour for neighbours in
                          extract_5x5(counting_grid, counting_index).flatten()
                          for neighbour in neighbours]
    dists = []
    dists2 = []
    for index in neighbour_indicies:
        dists.append([torus_dist(point, X[index]), index])

    if particle:
        if not All:
            dists = sorted(dists)[1:1 + n]
            indicies = [pair[1] for pair in dists]
            dists = [pair[0] for pair in dists]
        else:
            dists = sorted(dists)[1:]
            indicies = [pair[1] for pair in dists]
            dists = [pair[0] for pair in dists]
    else:
        if not All:
            dists = sorted(dists)[:n]
            indicies = [pair[1] for pair in dists]
            dists = [pair[0] for pair in dists]
        else:
            dists = sorted(dists)[:]
            indicies = [pair[1] for pair in dists]
            dists = [pair[0] for pair in dists]

    return indicies, dists


def psi_j(point):

    neighbour_indicies, _ = nearest_n_neighbours(point, n=6, particle=True)
    return (1 / 6) * sum([np.exp(1j * 6 * torus_angle(point, X[neighbour])) for neighbour in neighbour_indicies])


def local_disorder(point):

    neighbour_indicies, _ = nearest_n_neighbours(point, n=6, particle=True)

    angles = []
    for neighbour in neighbour_indicies:
        angles.append(torus_angle(point, X[neighbour]))
    angles = sorted(angles)
    angle_diffs = []
    for i in range(len(angles)):
        if i + 1 < len(angles):
            angle_diffs.append(angles[(i + 1)] - angles[i])
        else:
            angle_diffs.append(angles[0] - angles[i] + 2 * pi)
    return (np.var(angle_diffs))


def psi_6():
    return (1 / N) * sum([psi_j(point for point in X)])


def orientation_colormap(resolution):

    psis = [psi_j(point) for point in X]
    # print(psis)

    color_map = np.zeros((resolution, resolution))

    for xi, x in zip(range(resolution), np.linspace(0, Lx, resolution)):
        for yi, y in zip(range(resolution), np.linspace(0, Ly, resolution)):

            neighbour_indicies, dists = nearest_n_neighbours(
                (x, y), n=None, All=True, particle=False)

            weights = [1 / d for d in dists]
            weights = np.array(weights) / sum(weights)
            angle = np.angle(sum([weight * psis[index] for index,
                                  weight in zip(neighbour_indicies, weights)]), deg=True)

            color_map[xi, yi] = angle  # + int(bool(np.sign(angle)-1))*360
    return color_map


def disorder_colormap(resolution):

    disorders = [local_disorder(point) for point in X]

    color_map = np.zeros((resolution, resolution))

    for xi, x in zip(range(resolution), np.linspace(0, Lx, resolution)):
        for yi, y in zip(range(resolution), np.linspace(0, Ly, resolution)):

            neighbour_indicies, dists = nearest_n_neighbours(
                (x, y), n=None, All=True, particle=False)

            weights = [1 / d for d in dists]
            weights = np.array(weights) / sum(weights)
            disorder = sum([weight * disorders[index] for index,
                            weight in zip(neighbour_indicies, weights)])

            color_map[xi, yi] = disorder  # + int(bool(np.sign(angle)-1))*360

    return color_map


def metropolis_step(X):
    '''For a given state vector X, apply MC Metropolis move and update
        bookkeeping associated with counting grid'''
    global accepted, rejected, counting_grid

    # Original point
    index = fastrand.pcg32bounded(N)
    point_old = X[index].copy()
    counting_index_old = (int(point_old[1] * grid_marks_y / Ly),
                          int(point_old[0] * grid_marks_x / Lx))  # the index that the particle appears in in the counting grid

    # Propose new move
    point_new = [0, 0]
    point_new[0] = (point_old[0] + step * (fastrand.pcg32() / 2147483647)) % Lx
    point_new[1] = (point_old[1] + step * (fastrand.pcg32() / 2147483647)) % Ly
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


def g(dists, bins, r_range):
    global rings, first_run, dist_array

    l0 = min(Lx, Ly) / 2
    d0 = min(l0, r_range[1] * r)

    hist = np.histogram(dists, bins=bins, range=r_range)
    hist[0][0] = 0

    if first_run:
        circle_areas = pi * hist[1]**2
        rings = np.array([circle_areas[i + 1] - circle_areas[i]
                          for i in range(len(circle_areas) - 1)]) * N / (Lx * Ly)
        dist_array = hist[1][: -1] / r
        first_run = False

    g_of_r = hist[0] / rings
    return g_of_r, dist_array


def simulate(iters,
             eta,
             radial_dist=False,
             r_range=None,
             bins=None,
             plot_X0=False,
             plot_X1=False,
             plot_rd=False,
             poly_order=None):
    '''A subroutine for running the entire simulation. Everything handled'''
    global N, accepted, rejected, step, first_run

    # Initialise variables
    first_run = True
    accepted = 0
    rejected = 0
    X0 = init_lattice_eta(eta)
    init_grid(X0)
    step = d - 2 * r
    X = X0.copy()

    j = 0
    t0 = time.time()

    # In the case where we wish to compute a randial distribution
    if radial_dist:

        distribution_list = []

        while j < iters:
            X = metropolis_step(X)
            j += 1

            if j > 50000 and j % 1000 == 0:
                dist_array = squareform(pdist(X, torus_dist))
                for dists in dist_array:
                    radial_distribution, ticks = g(dists, bins=bins, r_range=r_range)
                    distribution_list.append(radial_distribution)

        mean_distribution = np.mean(distribution_list, axis=0)

    else:
        mean_distribution, ticks = None, None
        while j < iters:
            X = metropolis_step(X)
            j += 1

    # Print time/other stats
    print('Time taken: {0:.2f} s for {1} iterations'.format(time.time() - t0, iters))
    print('Acceptance Rate: {:.0f}%'.format(((accepted / (accepted + rejected)) * 100)))

    # Plots
    if plot_X0:
        plot_circles(X0)
    if plot_X1:
        plot_circles(X)
    if plot_rd:
        plot_radial_dist(ticks, mean_distribution, poly_order)

    return X, mean_distribution, ticks


if __name__ == '__main__':

    r = 0.02
    N = 1024

    for eta in [0.68]:

        accepted = rejected = 0
        X = init_lattice_eta(eta)
        init_grid(X)

        step = d - 2 * r
        for j in range(5000000):
            X = metropolis_step(X)
            # if 2000000 < j < 2040000 and j % 2000 == 0:
            # plot_circles(X,
            #              title=str(eta),
            #              circles=True,
            #              label=False,
            #              coloured_index=None,
            #              alpha=0.5,
            #              fill=False,
            #              LOP=True,
            #              resolution=40,
            #              gridlines=False)

        plot_circles(X,
                     title=str(eta),
                     circles=True,
                     label=False,
                     coloured_index=None,
                     alpha=0.5,
                     fill=False,
                     LOP=True,
                     LDP=False,
                     resolution=80,
                     gridlines=False)
        # lds = []
        # for n in range(256):
        #     lds.append(local_disorder(X[n]))
        # print(np.mean(lds))

        plt.savefig('{0}.png'.format(j - 4))

    plt.show()


plt.close('all')
