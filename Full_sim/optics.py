import matplotlib.pyplot as plt
import numpy as np
import math

N = 100 
radius = 0.25 #500 mm across
dx = (2 * radius)/N
edge_taper = 0.01 #gaussian
theta = math.pi / 4 #45 degrees (but in rad)
lamda = 3e-3 #3mm
frame_size = [50,50]

#tilted mirror matrices
# phase_gradient = np.zeros((3,1))
# degree_tilt = np.array([[1,0,0], [0, math.cos(theta), math.sin(theta)], [0, -math.sin(theta), math.cos(theta)]])
# coord = np.array([[x], [y], [z]])
# #QUESTION
mirror_coord = np.array([4,6,4]) #m

# phase_gradient = (degree_tilt * coord) + mirror_coord
phase_gradient = 1

def ap_grid(N, dx,radius, edge_taper, phase_gradient):
    coords = np.linspace(-radius, radius -dx, N)
    x, y = np.meshgrid(coords, coords)
    z = 0

    aperture = np.zeros(shape=(N,N), dtype=complex)
    r = np.sqrt(x**2 + y**2)

    #edge taper
    alpha = -np.log(edge_taper)/radius**2

    #Creates circular aperture
    inside = r < radius
    aperture[inside] = np.exp(-alpha * r[inside] ** 2)

    return aperture, x, y, z #check output for complex output with xyz values

def mirror(N, dx, theta, mirror_coord):
    length = ((mirror_coord[0]))/2
    width = ((mirror_coord[1]))/2
    distance = (mirror_coord[2])
    
    #account for non-square
    N_x = N 
    N_y = int(N * (width/length))

    length = np.linspace(-length, length, N_x)
    width = np.linspace(-width, width, N_y)

    x_grid, y_grid = np.meshgrid(length,width)

    mirror_data = [x_grid, y_grid, distance]

    return mirror_data, length, width, distance


def fresnel(aperture, mirror_data, lamda):
    # k = 2*np.pi / lamda
    # distance = float(mirror_data[2])
    # near_field = np.zeros(shape=(N,N), dtype=complex)
    
    # #create an observational grid (so as to not skew the)
    # obs_x = np.linspace(-mirror_data[0], mirror_data[0], N)
    # obs_y = np.linspace(-mirror_data[1], mirror_data[1], N)
    
    # for i in range(len(obs_x)):
    #     for j in range(len(obs_y)):

    #         #2d array of distances
    #         d = np.sqrt((x -obs_x[i])**2 + (y -obs_y[j])**2 + distance**2)
    #         near_field[i,j] = np.sum(aperture * np.exp(1j * k * d) / d, dtype=complex)

    # return near_field
    k = 2*np.pi / lamda
    dist = float(mirror_data[2])
    
    # Use the actual dimensions from the mirror function
    # instead of the global N
    target_x = length  # This is the 'length' array from mirror()
    target_y = width   # This is the 'width' array from mirror()
    
    # Initialize array with (rows, columns) -> (Y, X)
    near_field = np.zeros(shape=(len(target_y), len(target_x)), dtype=complex)
    
    # Loop over width (j/y) and length (i/x)
    for j in range(len(target_y)):
        for i in range(len(target_x)):
            # Distance from all aperture points (x, y) to ONE mirror point
            d = np.sqrt((x - target_x[i])**2 + (y - target_y[j])**2 + dist**2)
            
            # near_field[row, col]
            near_field[j, i] = np.sum(aperture * np.exp(1j * k * d) / d)

    return near_field



#def fraunhofer(a_grid, m_grid, lamda):


aperture, x, y, z = ap_grid(N, dx,radius, edge_taper, phase_gradient)
# plt.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
# plt.xlabel("Diameter of Aperture (mm)")
# plt.title("Aperture")
# plt.show()

mirror_data, length, width, distance = mirror(N, dx, theta, mirror_coord)


near = fresnel(aperture, mirror_data, lamda)
plt.xlabel("Length (m)")
plt.ylabel("Width (m)")
plt.title(f"Mirror Projection(Fresnel) Distance {distance} (m)")
plt.imshow(np.log10(np.abs(near)**2), extent=[length.min(), length.max(), width.min(), width.max()], aspect = 'equal')
plt.show()
