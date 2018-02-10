import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial.distance import pdist, squareform
import time
import fastrand
import matplotlib.cm
import seaborn as sns
from my_tools.time_ import Time_Block
from my_tools.misc import sigfig, beep
import multiprocessing

plt.close('all')
pi = np.pi
sqrt = np.sqrt

r = 0.02
N = 256


def angle(complex_number, deg=True):
    a = np.angle(complex_number, deg=deg)
    if a < 0:
        a += deg * 360 + (1 - deg) * 2 * pi
    return a


def axes_lims(x_data, y_data, gap_frac=0.1):
    if type(x_data) == tuple:
        x_min = min([min(x) for x in x_data])
        x_max = max([max(x) for x in x_data])
    else:
        x_min = min(x_data)
        x_max = max(x_data)
    if type(y_data) == tuple:
        y_min = min([min(y) for y in y_data])
        y_max = max([max(y) for y in y_data])
    else:
        y_min = min(y_data)
        y_max = max(y_data)

    x_diff = x_max - x_min
    y_diff = y_max - y_min

    y_lim = (y_min - gap_frac * y_diff, y_max + gap_frac * y_diff)
    x_lim = (x_min - gap_frac * x_diff, x_max + gap_frac * x_diff)

    return x_lim, y_lim


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


def extract_3X5(A, index):
    '''For a given index in an array, return the 3X5 matrix that surrounds
        that index, wrapping with torus topology at edges'''
    M3X5 = A.take(range(index[0] - 1, index[0] + 2), axis=0, mode='wrap')\
            .take(range(index[1] - 1, index[1] + 2), axis=1, mode='wrap')
    return M3X5


def extract_5x5(A, index):
    '''For a given index in an array, return the 5x5 matrix that surrounds
        that index, wrapping with torus topology at edges'''
    M5x5 = A.take(range(index[0] - 3, index[0] + 4), axis=0, mode='wrap')\
            .take(range(index[1] - 3, index[1] + 4), axis=1, mode='wrap')
    return M5x5


def g(dists, bins, r_range):
    global rings, first_run, dist_array

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


def grow_radii(eta0, eta1, mixing_steps=5000000, radius_steps=8, title='grown', plot_final=True, plot_growth=False):
    global r, accepted, rejected, X, step, first_run, step

    with Time_Block(block_name='grow radii function'):

        first_run = True
        accepted = 0
        rejected = 0
        X0 = init_lattice_eta(eta0)

        # final radius to be reached
        r0 = r
        eta_space = np.linspace(eta0, eta1, radius_steps)
        r_space = (eta_space * Lx * Ly / (N * pi))**0.5
        r1 = r_space[-1]

        init_grid(X0, grid_radius=r1)
        step = d - 2 * r
        X = X0.copy()

        with Time_Block(block_name='mixing steps'):

            for i in range(mixing_steps):
                X = metropolis_step(X)

        with Time_Block(block_name='overlap removal'):

            for i, R in enumerate(r_space[1:]):
                dr = abs(R - r)
                step = 6 * dr
                print(dr)
                r = R
                remove_overlaps()
                if plot_growth:
                    plot_circles(X, title='step {}'.format(i + 1), LOP=True, resolution=75)

        print('Eta = ', eta1, '\n', 'r = ' + sigfig(r, 3))

        if plot_final:
            plot_circles(X, title=title, LOP=True, resolution=250, overlaps=overlap_indicies)


def grow_radii_from_standard(number, eta1, radius_steps=8, title='grown', plot_final=True, plot_growth=False):
    global r, accepted, rejected, X, step, first_run, step, Lx, Ly, d, LX5, Ly2

    first_run = True
    accepted = 0
    rejected = 0
    X0 = np.load('X0{}.npy'.format(number))

    r = 0.02
    N = 256
    eta = 0.6
    eta0 = eta
    Lx = 0.78683555015768469
    Ly = 0.68141957503725981
    LX5 = Lx / 2
    Ly2 = Ly / 2
    d = Lx / sqrt(N)

    # final radius to be reached
    r0 = r
    eta_space = np.linspace(eta0, eta1, radius_steps)
    r_space = (eta_space * Lx * Ly / (N * pi))**0.5
    r1 = r_space[-1]

    init_grid(X0, grid_radius=r1)

    step = d - 2 * r
    X = X0.copy()

    # plot_circles(X, title='step 0', LOP=True, resolution=75)
    # plt.savefig('step 0.png')
    # plt.close()

    with Time_Block(block_name='overlap removal'):

        for i, R in enumerate(r_space[1:]):
            dr = abs(R - r)
            step = 6 * dr
            r = R
            remove_overlaps()
            if plot_growth:
                plot_circles(X, title='step {}'.format(i + 1), LOP=True, resolution=200)
                plt.savefig('step {}.png'.format(i + 1), dpi=300)
                plt.close()
    print('Eta = ', eta1, '\n', 'r = ' + sigfig(r, 3))

    if plot_final:
        plot_circles(X, title='step 20', LOP=True, resolution=75, overlaps=overlap_indicies)
        plt.savefig('step 20.png')
        plt.close()


def hex_7(point1, theta):
    points = [point1]
    thetas = [theta + i * pi / 3 for i in range(6)]
    for the in thetas:
        x = point1[0] + d * np.cos(the)
        y = point1[1] + d * np.sin(the)
        points.append([x, y])
    return points


def init_grid(X, grid_radius=None):
    '''Set up the initial grid for particle bookkeeping'''
    global grid_marks_x, grid_marks_y, counting_grid

    # Set optimal grid spacing
    if grid_radius == None:
        grid_marks_x = int(Lx / (2 * r))
        grid_marks_y = int(Ly / (2 * r))
    else:
        grid_marks_x = int(Lx / (2 * grid_radius))
        grid_marks_y = int(Ly / (2 * grid_radius))

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
    global Lx, Ly, d, LX5, Ly2, box_dims

    # Confirm N is a square number
    if sqrt(N) - int(sqrt(N)) != 0:
        raise Exception("N must be a square number")

    # Compute Lx and Ly variables
    k = sqrt(3) / 2
    Lx = sqrt(N * (r**2) * pi / (eta * k))
    Ly = sqrt(k * N * (r**2) * pi / eta)
    LX5 = Lx / 2
    Ly2 = Ly / 2
    box_dims = (Lx, Ly)

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
    print('N = {0}, Eta = {1:.4f}'.format(N, (N * pi * r**2) / (Lx * Ly)))
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


def metropolis_step(X, move_index=None):
    '''For a given state vector X, apply MC Metropolis move and update
        bookkeeping associated with counting grid'''
    global accepted, rejected, counting_grid

    # Original point
    if move_index == None:
        index = fastrand.pcg32bounded(N)
    else:
        index = move_index

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
                          extract_3X5(counting_grid, counting_index_new).flatten()
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


def metropolis_step_removal(X, move_index=None):
    '''For a given state vector X, apply MC Metropolis move and update
        bookkeeping associated with counting grid'''
    global accepted, rejected, counting_grid

    # Original point
    if move_index == None:
        index = fastrand.pcg32bounded(N)
    else:
        index = move_index

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
                          extract_3X5(counting_grid, counting_index_new).flatten()
                          for neighbour in neighbours]

    # Determine whether overlap occurs
    # Here is the difference from old step. Move accepted if it's further away
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


def nearest_n_neighbours(point, n=None, All=False, particle=True):
    '''For a given particle or (x,y) coordinate, return the indicies and
        distances of the n nearest neighbours. Note, n cannot be larger than about 10'''
    # if type(point) != tuple:
    #     print(point, list(point), type(point))
    counting_index = (int(point[1] * grid_marks_y / Ly),
                      int(point[0] * grid_marks_x / Lx))

    neighbour_indicies = [neighbour for neighbours in
                          extract_5x5(counting_grid, counting_index).flatten()
                          for neighbour in neighbours]

    n_neighbours = len(neighbour_indicies)

    dists = np.zeros((n_neighbours, 2))
    for i, index in enumerate(neighbour_indicies):
        dists[i, :] = torus_dist(point, X[index]), index

    dists = dists[np.argsort(dists[:, 0])]

    if All:
        n = n_neighbours

    if particle:
        # indicies, distances
        return (dists[1:n + 1, 1]).astype(int), dists[1:n + 1, 0]
    else:
        # indicies, distances
        return (dists[:n, 1]).astype(int), dists[:n, 0]


def neighbour_indicies_3X5(central_point):
    '''Central point is tuple of x,y coords of point of interest.
        indicies of neigbouring points in 3X5 gird returned'''

    grid_index = (int(central_point[1] * grid_marks_y / Ly),
                  int(central_point[0] * grid_marks_x / Lx))

    neighbour_indicies = [neighbour for neighbours in
                          extract_3X5(counting_grid, grid_index).flatten()
                          for neighbour in neighbours]

    return neighbour_indicies


def neighbour_indicies_5x5(central_point):
    '''Central point is tuple of x,y coords of point of interest.
        indicies of neigbouring points in 3X5 gird returned'''

    grid_index = (int(central_point[1] * grid_marks_y / Ly),
                  int(central_point[0] * grid_marks_x / Lx))

    neighbour_indicies = [neighbour for neighbours in
                          extract_5x5(counting_grid, grid_index).flatten()
                          for neighbour in neighbours]

    return neighbour_indicies


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
            color_map[xi, yi] = angle(sum([weight * psis[index] for index,
                                           weight in zip(neighbour_indicies, weights)]), deg=True)

    return color_map


def plot_circles(X,
                 title='',
                 circles=True,
                 label=False,
                 coloured_index=None,
                 overlaps=[],
                 alpha=1,
                 fill=False,
                 LOP=False,
                 LDP=False,
                 resolution=100,
                 gridlines=True):
    '''Plot particle positions in box'''

    fig, ax = plt.subplots()

    # plot Cirlces
    if circles:
        def crosses_boundary(p):
            '''Determine whether particle crosses the boundaries for neat plotting'''
            cross_x_r = p[0] + r > Lx  # cross x on right?
            cross_x_l = p[0] - r < 0  # cross x on left?
            cross_y_t = p[1] + r > Ly  # cross y on top?
            cross_y_b = p[1] - r < 0  # cross y on bottom?
            return cross_x_r, cross_x_l, cross_y_t, cross_y_b

        if coloured_index != None:
            neighbour_indicies, _ = nearest_n_neighbours(
                X[coloured_index], n=6, All=False, particle=True)
        else:
            neighbour_indicies = []

        for index, pair in enumerate(X):

            # Add a single blue particle to get sense of movement
            if index == coloured_index:
                c = 'k'
            elif index in neighbour_indicies:
                c = 'k'
            else:
                c = 'k'

            if index in overlaps:
                c = 'red'

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
                        (pair[0] + Lx, pair[1] + Ly), r, color=c, alpha=alpha, fill=fill))
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

        vim = 0
        vmax = 360
        cm = orientation_colormap(resolution=resolution)

        im1 = plt.imshow(cm,
                         extent=(0, Lx, 0, Ly),
                         origin='lower',
                         cmap=matplotlib.cm.hsv,
                         vmin=vim,
                         vmax=vmax,
                         alpha=0.8)

        cb = plt.colorbar()
        cb.set_label('arg($\psi_j$), degrees', rotation=270, labelpad=20)

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

    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.title(title)


def plot_radial_dist(x_ticks, radial_dist, poly_order=5, y_err=None, plot_rd=False):
    '''Function that scatters all points recorded for the radial distribution
        function with a polynomial fit'''

    xp = np.linspace(x_ticks[0], x_ticks[-1], 5000)

    x_ticks = x_ticks[np.where(x_ticks > 2)]
    radial_dist = radial_dist[np.where(x_ticks > 2)]

    if y_err != None:
        y_err = y_err[np.where(x_ticks > 2)]

    poly_func = np.poly1d(np.polyfit(x_ticks[1:], radial_dist[1:], poly_order))

    if plot_rd:
        fig, ax = plt.subplots()

        plt.scatter(x_ticks, radial_dist, color='r', alpha=0.8)
        plt.plot(xp, poly_func(xp), color='k')

        plt.xlabel('Radial Speration')
        plt.ylabel('Distribution Function')
        title = 'N = {0:.0f}, Eta = {1:.3f}'.format(
            N, (N * pi * r**2) / (Lx * Ly))

        x_lims, y_lims = axes_lims(x_ticks, radial_dist)
        ax.set_ylim(y_lims)
        plt.title(title)

        plt.errorbar(x_ticks, radial_dist, yerr=y_err, fmt=None,
                     ecolor='k', linewidth='0.7', capthick=1)
    # print('Eta: {0:.3f}, g(2r): {1}'.format(eta, p(2)))
    return poly_func


def pressure(X0, iters, r_range, bins=50, sample_rate=5, plot_rd=True):

    global accepted, rejected, step, X, first_run, a

    first_run = True
    init_grid(X0)
    X = X0.copy()
    step = d - 2 * r

    sample_iter_rate = N * sample_rate
    master_distribution_list = []

    for j in range(iters):
        X = metropolis_step(X)

        if j % sample_iter_rate == sample_iter_rate - 1:
            for x in X:
                neighbour_indicies = neighbour_indicies_3X5(x)
                dists = [torus_dist(x, X[neighbour]) for neighbour in neighbour_indicies]
                radial_distribution, ticks = g(dists, bins=bins, r_range=r_range)
                master_distribution_list.append(radial_distribution)

    master_distribution_list = np.array(master_distribution_list)
    mean_distribution = np.mean(master_distribution_list, axis=0)
    uncertainty = np.std(master_distribution_list, axis=0) / \
        sqrt(np.shape(master_distribution_list)[0])
    poly_func = plot_radial_dist(ticks, mean_distribution, poly_order=1, plot_rd=plot_rd)

    return poly_func(2)


def psi_6():
    return (1 / N) * sum([psi_j(point) for point in X])


def psi_j(point):

    neighbour_indicies, _ = nearest_n_neighbours(point, n=6, particle=True)
    return (1 / 6) * sum([np.exp(1j * 6 * torus_angle(point, X[neighbour])) for neighbour in neighbour_indicies])


def remove_overlaps():
    '''Takes global variable X, containing overlaps, and attempts to remove them'''
    global X, overlap_indicies

    #
    max_iters = 5000
    overlap_indicies = []

    for index, x in enumerate(X):
        neighbour_indicies = neighbour_indicies_3X5(x)
        dists = np.array([torus_dist(x, X[neighbour])
                          for neighbour in neighbour_indicies])
        overlaps = (dists != 0) & (dists < 2 * r)
        if np.any(overlaps):
            overlap_indicies.append(index)

    # plot_circles(X, overlaps=overlap_indicies, title='before')

    for i in range(max_iters):
        overlapper_index = overlap_indicies[i % len(overlap_indicies)]
        overlapper = X[overlapper_index]
        neighbour_indicies = neighbour_indicies_3X5(overlapper)
        for j in range(100):
            neighbour_index = neighbour_indicies[i % len(neighbour_indicies)]
            X = metropolis_step(X, move_index=neighbour_index)
            if j % 10 == 9:
                dists = np.array([torus_dist(overlapper, X[neighbour])
                                  for neighbour in neighbour_indicies])
                overlaps = (dists != 0) & (dists < 2 * r)
                if np.any(overlaps):
                    pass
                else:
                    overlap_indicies.remove(overlapper_index)
                    break

        if len(overlap_indicies) == 0:
            return

        if i == max_iters - 1:
            print('ATTEMPT 1 FAILED')

    for i in range(max_iters):
        overlapper_index = overlap_indicies[i % len(overlap_indicies)]
        overlapper = X[overlapper_index]
        neighbour_indicies = neighbour_indicies_3X5(overlapper)
        for j in range(100):
            neighbour_index = neighbour_indicies[i % len(neighbour_indicies)]
            X = metropolis_step(X, move_index=neighbour_index)
            if j % 10 == 9:
                dists = np.array([torus_dist(overlapper, X[neighbour])
                                  for neighbour in neighbour_indicies])
                overlaps = (dists != 0) & (dists < 2 * r)
                if np.any(overlaps):
                    pass
                else:
                    overlap_indicies.remove(overlapper_index)
                    break

        if len(overlap_indicies) == 0:
            print('ATTEMPT 2 SUCCESS')
            return

        if i == max_iters - 1:
            print('ATTEMPT 2 FAILED')
        # plot_circles(X, overlaps=overlap_indicies, title='after')


def rotate(v, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(rotation_matrix, v)


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
        plot_radial_dist(ticks, mean_distribution, poly_order, plot_rd=True)

    return X, mean_distribution, ticks


def torus_angle(ref_point, other_point):

    dx = other_point[0] - ref_point[0]
    dy = other_point[1] - ref_point[1]

    if abs(dx) > LX5:
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


def torus_dist(p1, p2):
    '''Outputs the closest distance between two points on a 2d
        plane with torus-like topology'''

    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])

    if dx > LX5:
        dx = Lx - dx
    if dy > Ly2:
        dy = Ly - dy

    return (dx**2 + dy**2)**0.5


def metastable_pressure(eta):
    grow_radii(0.6, eta)


def crystallisation(eta):
    grow_radii(0.6, eta, radius_)


def phase_diagram():

    data = True

    if data:

        etas = np.array([0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.695,
                         0.7,  0.705, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76])

        g_values = np.array([5.1719877525, 5.4184515271, 5.6611982324, 5.9016309346,
                             6.1950132936, 6.4228699121, 6.6221753384,
                             6.3752722856,  6.1215568825,
                             5.9008810983, 5.9527052689, 6.1669155068, 6.4249247868,
                             6.7266886272, 7.07919022, 7.51])

        g_uncert = np.array([0.0093708043, 0.0116720653, 0.0131396732, 0.0106090006,
                             0.0095224986, 0.0155700962, 0.0445249236,
                             0.0918062541,  0.0302415387,
                             0.0165904105, 0.011786573, 0.0115487812, 0.0123262563,
                             0.0097955388, 0.0103623641, 0.012])

        meta_etas = np.array([0.69, 0.695, 0.7, 0.705, 0.71, 0.715, 0.72, 0.73, 0.74, 0.75])

        meta_gs = np.array([10.13860196 + 3.3, 13.8879075608, 14.3779824976, 14.8017211956, 15.8718666257,
                            16.4283628758, 17.1828657845, 18.737948506, 20.4701858224,
                            22.8241352137]) - 3.3

        meta_uncs = np.array([0.1957489542, 0.1984789966, 0.2914987512, 0.231409555,
                              0.2485494228, 0.3913291465, 0.3147388692, 0.345488746,
                              0.379844645])

        Z_values = 1 + 2 * etas * g_values
        Z_uncert = 2 * etas * g_uncert

    fig = plt.figure(figsize=(7, 9))
    ax = plt.gca()

    stable_x = etas[1:]
    stable_y = Z_values[1:]
    meta_x = meta_etas
    meta_y = meta_gs

    plt.scatter(stable_x,
                stable_y,
                marker='x',
                s=80,
                alpha=0.9,
                color='k',
                linewidth=0.9,
                label='Stable',)
    ax.errorbar(stable_x,
                stable_y,
                yerr=Z_uncert[1:],
                alpha=0.9,
                fmt='none',
                color='k',
                capsize=2,
                capthick=0.8,
                linewidth=0.9,
                elinewidth=0.9)
    plt.plot(stable_x,
             stable_y,
             color='k',
             alpha=0.7,
             linewidth=0.8,
             linestyle='--')

    plt.scatter(meta_etas[1:],
                meta_gs[1:],
                marker='x',
                alpha=0.9,
                s=80,
                color='firebrick',
                linewidth=0.8,
                label='Metastable',)
    ax.errorbar(meta_etas[1:],
                meta_gs[1:],
                yerr=meta_uncs,
                fmt='none',
                color='firebrick',
                alpha=0.9, capsize=2,
                capthick=0.8,
                linewidth=0.8,
                elinewidth=0.8)
    plt.plot(meta_etas,
             meta_gs,
             linewidth=0.8,
             linestyle='--',
             color='r',
             alpha=0.5)

    y_lims = (7.32, 21.4)
    x_lims = (0.63, 0.77)
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    stem_x = [stable_x[0], stable_x[5], stable_x[8], stable_x[-1]]
    stem_y = [stable_y[0], stable_y[5], stable_y[8], stable_y[-1]]

    for line_x, line_y in zip(stem_x, stem_y):
        ax.axvline(x=line_x,
                   ymax=(line_y - y_lims[0]) / (y_lims[1] - y_lims[0]),
                   linestyle='--',
                   color='k',
                   alpha=0.5,
                   linewidth=0.9)

    plt.annotate('Liquid Phase', xy=(0.655, 7.8))
    plt.annotate('Coexistence', xy=(0.687, 8.6))
    plt.annotate('Solid Phase', xy=(0.725, 8.1))
    plt.annotate('Metastable \n   Branch', xy=(0.73, 14.3))

    plt.xlabel('Packing Fraction, $\eta$')
    plt.ylabel('Compressibility Factor, $Z$')
    plt.title('Hard Disk Equation of State')

    axbox = ax.get_position()
    plt.legend(loc=1)

    # plt.savefig('EOS.png', format='png', dpi=300)
    plt.show()


def metastable_pressure(number, eta1, iters, bins=25, sample_rate=5, plot_rd=True):
    global X, ticks, mean_distribution, master_distribution_list, radial_distribution

    grow_radii_from_standard(number, eta1, radius_steps=20, plot_final=False, plot_growth=False)
    master_distribution_list = []

    r_range = (2 * r, 2.1 * r)

    for i in range(iters):
        X = metropolis_step(X)
        if i % int(iters / 20) - 1 == 0:
            for x in X:
                neighbour_indicies = neighbour_indicies_3X5(x)
                dists = [torus_dist(x, X[neighbour]) for neighbour in neighbour_indicies]
                radial_distribution, ticks = g(dists, bins, r_range)
                master_distribution_list.append(radial_distribution)

    master_distribution_list = np.array(master_distribution_list)
    mean_distribution = np.mean(master_distribution_list, axis=0)
    uncertainty = np.std(master_distribution_list, axis=0) / \
        sqrt(np.shape(master_distribution_list)[0])

    poly_func = plot_radial_dist(ticks, mean_distribution, poly_order=1, plot_rd=plot_rd)

    print('\n', 'pressure: {}'.format(1 + 2 * eta1 * poly_func(2)), '\n')
    return poly_func(2)


def autocorrelation1(steps, points):
    global X
    steps_per_point = int(steps / points)
    psis = []
    for j in range(steps):
        X = metropolis_step(X)
        if j % steps_per_point == 0:
            psis.append(psi_6())
    psis = np.array(psis)
    p1 = psis[:-1]
    p2 = np.conj(psis[1:])
    C = (p1 * p2) / np.absolute(p1)
    return C


def autocorrelation2(steps, points):
    global X
    steps_per_point = int(steps / points)
    psis = []
    for j in range(steps):
        X = metropolis_step(X)
        if j % steps_per_point == 0:
            psis.append(psi_6())
    psis = np.array(psis)
    C = np.angle(psis)
    return C


if __name__ == '__main__':

    # grow_radii_from_standard(3, 0.74, radius_steps=20, plot_growth=True)
    # phase_diagram()

    # phase_diagram()

    # first_run = True
    # accepted = 0
    # rejected = 0
    # X0 = init_lattice_eta(0.73)
    # init_grid(X0)
    # step = d - 2 * r
    # X = X0.copy()
    #
    # with Time_Block(block_name='standard metropolis'):
    #     for i in range(5000000):
    #         X = metropolis_step(X)

    grow_radii_from_standard(3, 0.74, radius_steps=20, plot_growth=False)
    points = 1000
    steps = 5000000
    c = autocorrelation2(steps, points)
    #plt.plot(list(tange(points)), c)
    plt.plot(np.arange(points), np.real(c))
    plt.show()
    # plot_circles(X, LOP=True, resolution=200)
    # plt.savefig('0.png', format='png', dpi=300)

    # photos = (np.linspace(0, 1410, 20)**2).astype(int)
    # step = d - 2 * r
    # for i in range(2000000):
    #     X = metropolis_step(X)
    #     if i in photos:
    #         plot_circles(X, LOP=True, resolution=100)
    #         plt.savefig('runtwelve{}.png'.format(i), format='png', dpi=300)
    #         # beep()

    # first_run = True
    # accepted = 0
    # rejected = 0
    # X = np.load('X00.npy')
    #
    # r = 0.02
    # N = 256
    # eta = 0.6
    # Lx = 0.78683555015768469
    # Ly = 0.68141957503725981
    # LX5 = Lx / 2
    # Ly2 = Ly / 2
    # d = Lx / sqrt(N)
    # init_grid(X, grid_radius=0.02)

    # np.save('X00', X)

    # number = 9
    # print('DATA FROM X0{}'.format(number))
    # metastable_pressure(number, 0.75, 500000)

    # plot_circles(X, LOP=True, resolution=75)

    #     plot_circles(X, title='From Lattice', LOP=True)

    # total_iters = 500000
    # pressure_steps =
    # iters_per_step = int(total_iters / pressure_steps)
    # ps = []
    # plt.savefig('init')
    # with Time_Block():
    #     for i in range(pressure_steps):
    #         p = pressure(X, iters_per_step, (2 * r, 2.1 * r), plot_rd=False)
    #         print('pressure: ', p)
    #         plot_circles(X, title='step {0}, pressure = {1}'.format(i, sigfig(p, 3)))
    #         plt.savefig('fig{}'.format(i))
    #         ps.append(p)

    # with Time_Block():
    #     plot_circles(X, title='0')
    #     plt.savefig('0.png', format='png')
    #     breaks = 20
    #     steps = 600000
    #     break_steps = int(steps / breaks)
    #     for i in range(steps):
    #         X = metropolis_step(X)
    #         if i % (break_steps - 1) == 0:
    #             plot_circles(X, title='{}'.format(i + 1))
    #             plt.savefig('{}.png'.format(i + 1), format='png')

    # jobs = []
    # steps = [8, 10]
    # for s in steps:
    #     p = multiprocessing.Process(target=grow_radii, args=(0.6, 0.74), kwargs={
    #                                 'radius_steps': s, 'title': 'steps: {}'.format(s)})
    #     jobs.append(p)
    #     p.start()

    beep()
    # plt.show()
