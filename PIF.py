import numpy as np
import pygame as pg
import matplotlib as plt

def parameters():
    
    size = (1000,750) # mida de la simulació en píxels
    k = 2 # llavor de terreny, comprèn valors de 0 a 100. Si vols que sigui aleatòria utilitza k=np.random.randint(0,100).
    starting_point = (40,40) # posició d'inici de l'incendi
    
    shift = 1 # 0 si el vent és lineal, 1 si gira.
    wind_speed = 10 # velocitat del vent. Si vols que sigui aleatòria utilitza wind_speed = (12 - 1)*np.random.random() + 1
    
    wind_direction = 1. # direcció del vent en radiants. Si vols que sigui aleatòria utilitza wind_direction = (2*np.pi-0)*np.random.random() + 0
    wind_shift = 0.5 # desviació del vent en radiants. Si vols que sigui aleatòria utilitza wind_shift = (np.pi/4 - (-1)*np.pi/4)*np.random.random()
    
    print("Llavor = ",k)
    print("Posició inicial de l'incendi = ",starting_point)
    print("Velocitat del vent = ",wind_speed," m/s")
    print("Direcció del vent = ", wind_direction," rad")
    if shift==1:
        print("Desviació del vent = ", wind_shift," rad")

    
    return size, k, starting_point, shift, wind_speed, wind_direction, wind_shift

def main():
    
    RED = (255,0,0)
    GRAY = (71,75,78)
    parametres = parameters()
    sizex,sizey = parametres[0]
    
    step = 10
    
    k = parametres[1]
    
    inicifoci,inicifocj = parametres[2]
    
    shift = parametres[3]
    speed = parametres[4]
    theta = parametres[5]
    theta_hat = parametres[6] + theta
    
    

    dimm = sizex//step
    dimn = sizey//step

    physical_data, burning_iterations = get_physical_data(dimn, dimm, k)
    wind_speed, mean_speed = generate_wind_speed(dimn, dimm, speed)
    wind_dir = generate_wind_direction(dimn, dimm, shift, theta, theta_hat)
    wind_data = get_wind_data(wind_speed, wind_dir, mean_speed)
    propagation_data=get_propagation_data(physical_data,wind_data)

    
    # VISUALIZE VECTOR FIELD
    
    xt = np.linspace(0, dimn, 10)
    yt = np.linspace(0, dimm, 10)

    wind_direction = wind_dir.tolist()
    utmesh = []
    vtmesh = []
    for i in range(10):
        for j in range(10):
            utmesh.append(wind_direction[i*dimn//10][j*dimm//10][0])
            vtmesh.append(wind_direction[i*dimn//10][j*dimm//10][1])

    xtmesh, ytmesh = np.meshgrid(xt, yt)

    plt.pyplot.quiver(xtmesh, ytmesh, utmesh, vtmesh)

    plt.pyplot.show()
    
    
    
    
    pg.init()

    screen = interface(physical_data,sizex,sizey,step)

    colors = np.zeros((dimn,dimm),int)        
    colors[inicifoci][inicifocj] = 1
    rect = pg.draw.rect(screen, RED, [inicifocj*step+1,inicifoci*step+1,step-2,step-2], 1)
    pg.Surface.fill(screen, RED, rect)
    
    done = False
    clock = pg.time.Clock()
    frame = np.zeros((dimn,dimm))

    while not done:
        
        for e in pg.event.get():  # User did something
            if e.type == pg.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            else:
                if e.type == pg.MOUSEWHEEL:
                    bound = np.random.randint(0,100)/100
                    
                    next_colors = colors.astype(int)
                    for i in range(0,dimn):   
                        for j in range(0,dimm):
                            
                            if colors[i][j] == 1:
                                if frame[i][j] >= burning_iterations[i][j]+2 or frame[i][j] == 10:
                                    rect = pg.draw.rect(screen, GRAY, [j*step+1, i*step+1, step-2, step-2], 1)
                                    pg.Surface.fill(screen, GRAY, rect)
                                    colors[i][j] = 2
                                else:
                                    frame[i][j] += 0.05
                                    neighbourhood = get_neighbour_cells(dimn, dimm, i, j)
                                    for neighbour in neighbourhood:                                        
                                        propagation = update(propagation_data, (i,j), neighbour)
                                        
                                        if colors[neighbour[0]][neighbour[1]] == 0 and propagation >= bound:
                                            
                                            rect = pg.draw.rect(screen, RED, [neighbour[1]*step+1,neighbour[0]*step+1,step-2,step-2],1)
                                            pg.Surface.fill(screen, RED, rect)
                                            next_colors[neighbour[0]][neighbour[1]] = 1                        
                            
                    pg.display.update()
                    colors = next_colors
                    
        pg.display.flip()
        clock.tick(60)
        
    pg.quit()
    

def perlin_noise(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

def generate_perlin_data(k, dimn, dimm, soft):
    p = np.zeros((dimn,dimm))
    for i in range(soft):
        freq = 2**i
        linx = np.linspace(0, freq, dimm, endpoint=False)
        liny = np.linspace(0, freq, dimn, endpoint=False)
        x, y = np.meshgrid(linx, liny)  # FIX3: I thought I had to invert x and y here but it was a mistake
        p = perlin_noise(y, x, seed=k) / freq + p
    
    pmax=p.max()
    pmin=p.min()
    p=(p-pmin)/(pmax-pmin)

    return p

def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y

def get_neighbour_cells(n, m, i, j):
    """
    Given a matrix, it gets the Moore neighbourhood of each cells.
    
    Args:
        - matrix: an (n,m) numpy matrix.
        - i: non-negative integer, row number from 0 to n-1.
        - j: non-negative integer, column number from 0 to m-1.
        
    Returns: a list L containing in L[i][j] a list of the matrix's indexes that
    represent the neighbourhood of cell [i][j] for every i, j.
    """
    indexes = []
    if i == 0:
        if j == 0:
            indexes = [[0, 1], [1, 1], [1, 0]]
        if j == (m-1):
            indexes = [[0, m-2], [1, m-2], [1, m-1]]
        if j in range(1,m-1):
            indexes = [[0, j-1], [0, j+1], [1, j-1], [1, j], [1, j+1]]
    if i == (n-1):
        if j == 0:
            indexes = [[n-2, 0], [n-2, 1], [n-1, 1]]
        if j == (m-1):
            indexes = [[n-1, m-2], [n-2, m-2], [n-2, m-1]]
        if j in range(1,m-1):
            indexes = [[n-1, j-1], [n-1, j+1], [n-2, j-1], [n-2, j], [n-2, j+1]]
    
    if i in range(1,n-1):
        if j == 0:
            indexes = [[i-1, 0], [i+1, 0], [i-1, 1], [i, 1], [i+1, 1]]
        if j == (m-1):
            indexes = [[i-1, m-1], [i+1, m-1], [i-1, m-2], [i, m-2], [i+1, m-2]]
        if j in range(1,m-1):
            indexes = [[i-1, j-1], [i, j-1], [i+1, j-1], [i-1, j], [i+1, j], [i-1, j+1], [i, j+1], [i+1, j+1]]
            
    return indexes


def get_physical_data(dimn, dimm, k):
    
    density_data = np.zeros((dimn,dimm))
    type_data = np.zeros((dimn,dimm))
    physical_data = np.zeros((dimn,dimm))
    burning_iterations = np.zeros((dimn,dimm))

    
    for i in range(2):    
        density_data = generate_perlin_data(k,dimn,dimm,8)
        
    for i in range(2):    
        type_data = generate_perlin_data(k,dimn,dimm,8)
    
    for i in range(dimn):
        for j in range(dimm):
            
            physical_data[i][j] = np.sqrt(density_data[i][j] * type_data[i][j])
            if type_data[i][j]==0:
                type_data[i][j]=0.01
            
            burning_iterations[i][j] = (density_data[i][j] / type_data[i][j])
    
    return physical_data, burning_iterations


def generate_wind_direction(n, m, shift, theta, theta_hat):
    """
    Generates the wind direction matrix from an initial seed as the wind speed
    that enters our grid and allows a pi/4 direction shift if shift = 1, and if
    shift = 0 it considers a stable wind direction throughout the terrain.
    
    Args:
        - theta: real-valued number from 0 to 2pi.
        - shift: takes 0 and 1 as values to indicate whether the wind direction
        is assumed constant or some shift is allowed.
    
    Returns: an (n,m,2) numpy array representing two-dimensional vectors as the
    wind directions for each cell in the CA.
    """
    wind_direction_tensor = np.zeros((n, m, 2))
    
    
    if shift == 0:
        wind_direction_tensor += np.array([np.cos(theta), np.sin(theta)])
    
    if shift == 1:
        theta_new = np.array([theta, theta])
        theta_update = np.array([(theta_hat-theta)/(m-1), (theta_hat-theta)/(n-1)])
        for i in range(n):
            for j in range(m):
                wind_direction_tensor[i][j] = np.array([np.cos(theta_new[0]), np.sin(theta_new[1])])
            theta_new += theta_update

    return wind_direction_tensor

def generate_wind_speed(n, m, mean_speed):
    

    """
    Given the size of the cellular automata, provides with a numerical description
    of the wind speed to each cell following a random mean speed and a normal
    distribution allowing a 10% of noise with respect to the mean value to 99.8%
    of the sample space, discarding outliers contaning more than a 15% of error,
    and a normalization of these data to smoothen transitions between adjacent cells. 
    
    Args:
        - n, m: non-negative integers indicating the (n,m) numpy matrix size.
    
    Returns: an (n,m) numpy matrix representing the coefficients for wind speed.
    """
    Wind = np.zeros((n,m))
    
    
    for i in range(n):
        for j in range(m):
            temp = 0
            while (temp < 0.85*mean_speed) or (temp > 1.15*mean_speed):
                temp = np.random.normal(mean_speed, 0.1/3*mean_speed)
            Wind[i][j] = temp
    
    wind_speed_matrix = Wind.copy()
    
    for i in range(n):
        for j in range(m):
            neighbourhood = get_neighbour_cells(Wind.shape[0], Wind.shape[1], i, j)
            normalized = 0
            for neighbour in neighbourhood:
                normalized += Wind[neighbour[0], neighbour[1]]
            normalized += Wind[i][j]
            wind_speed_matrix[i][j] = normalized/(len(neighbourhood)+1)
    
    return wind_speed_matrix,mean_speed

def wind_parameters():
    """
    Computes the parameters alpha and beta that best fit the data.
    
    Args:
        - wind_speed: (n,) numpy array, domain.
        - wind_speed_adjustment_factor: (n,) numpy array, image.
    
    Returns: alpha, beta, the two parameters representing the wind factor
    """
    wind_speed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    wind_adjustment_factor = np.array([1.2, 1.4, 1.7, 2, 2.4, 2.9, 3.3, 4.1, 5.1, 6, 7.3, 8.5])
    log_wind_adjustment_factor = np.log(wind_adjustment_factor)
    X = np.ones((12, 2))
    X[:,1] = wind_speed
    y = log_wind_adjustment_factor
    beta_hat = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))
    alpha = beta_hat[1]
    beta = np.exp(beta_hat[0])
    return alpha, beta

def get_wind_data(wind_speed_matrix, wind_direction_tensor, mean_speed):
    """
    Computes the wind factor given the wind speed matrix and the uniform wind
    direction of the fire propagation from the inducing cell to the induced cell,
    given the wind parameters that best fit the wind adjustment factors.
    """
    n, m = wind_speed_matrix.shape
    alpha, beta = wind_parameters()
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    wind_max = mean_speed*1.15
    
    wind_data = np.zeros((n, m, 3, 3))
    for i in range(n):
        for j in range(m):
            neighbourhood = get_neighbour_cells(n, m, i, j)
            for neighbour in neighbourhood:
                neighbour_coordinates = np.array([neighbour[0]-i, neighbour[1]-j])
                neighbour_vector = np.matmul(rotation_matrix, neighbour_coordinates)
                cosine_factor = np.dot(neighbour_vector, wind_direction_tensor[i][j])/(np.linalg.norm(neighbour_vector)*np.linalg.norm(wind_direction_tensor[i][j]))
                data_point = beta*np.exp(alpha*(wind_speed_matrix[i][j]*cosine_factor-wind_max))
                wind_data[i][j][1+neighbour_coordinates[0]][1+neighbour_coordinates[1]] = data_point
    print("alpha = ",alpha)
    print("beta = ",beta)
    print("cosine = ",np.dot(np.array([1,0]), wind_direction_tensor[0][0]))
    return wind_data


def get_propagation_data(physical_data, wind_data):
    """
    Computes the propagation function from inducing cell to induced cell.
    
    Args:
        - density_type_data: (n,m) numpy array containing the vegetation density
        data as the combustion configuration for each cell in the CA.
        - veg_type_data: (n,m) numpy array containing the vegetation type data
        as the combustion flammability for each cell in the CA.
        - wind_data: (n,m,3,3) numpy array containing the wind factor information
        of each neighbour of each neighbourhood of each cell in the CA.
    
    Returns: an (n,m,3,3) numpy array containing the propagation function
    information, and a (n,m,3,3) numpy array containing the propagation function
    information as probabilities for each neighbourhood.
    """    
    n, m = physical_data.shape
    propagation_data = np.zeros((n, m, 3, 3))
    for i in range(n):
        for j in range(m):
            propagation_data[i][j] = physical_data[i][j]*wind_data[i][j]
            
    return propagation_data

def update(propagation_data, inducing_cell, induced_cell):
    
    i=induced_cell[0]-inducing_cell[0]+1
    j=induced_cell[1]-inducing_cell[1]+1
    
    propagation_value=propagation_data[inducing_cell[0]][inducing_cell[1]][i][j]
    
    return propagation_value
       

def interface(physical_data,sizex,sizey,step):
    """
    Given the size of the interface and the physical data matrix, generates the grid with a 
    green gradient depending of the value of the vegetation matrix.
    
    """
    
    BLACK = (0,0,0)
    size = (sizex,sizey)
    screen = pg.display.set_mode(size)

    screen.fill(BLACK)

    pg.display.set_caption("Propagació d'incendis forestals")

    for i in range(0,sizex,step):
        for j in range(0,sizey,step):
            t = physical_data[j//step][i//step]*255
            rect = pg.draw.rect(screen, (0,t,0), [i+1,j+1,step-2,step-2],1)
            pg.Surface.fill(screen, (0,t,0), rect)
    return screen


main()