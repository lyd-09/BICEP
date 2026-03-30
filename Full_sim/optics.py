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
mirror_coord = np.array([2,2,4]) #m

# phase_gradient = (degree_tilt * coord) + mirror_coord
phase_gradient = 1

def aperture(N, dx,radius, edge_taper, phase_gradient):
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
    length = ((mirror_coord[0]) * 1000)/2 #mm (/2 to account for pos/negative sides)
    width = ((mirror_coord[1]) * 1000)/2 #mm
    distance = ((mirror_coord[2]) *1000)  #mm
    
    length = np.linspace(-length, length - dx, N)
    width = np.linspace(-width, width - dx, N)

    x_grid, y_grid = np.meshgrid(length,width)

    mirror = [x_grid, y_grid, distance]
    return mirror, length, width, distance


def fresnel(aperture, mirror, lamda):
    k = 2*np.pi / lamda
    near_field = np.zeros(shape=(N,N), dtype=complex)
    # f_length = np.linspace(-frame_size[0]/2, frame_size[0]/2, N)
    # f_width = np.linspace(-frame_size[1]/2, frame_size[1]/2, N)
    
    for i in range(len(length)):
        for j in range(len(width)):

            #2d array of distances
            d = np.sqrt((mirror[0] - length[i])**2 + (mirror[1] - width[j])**2 + distance**2)
            near_field[i,j] = np.sum(aperture * np.exp(1j *k * d) /d )

    return near_field

#def fraunhofer(a_grid, m_grid, lamda):


aperture, x, y, z = aperture(N, dx,radius, edge_taper, phase_gradient)
# plt.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
# plt.xlabel("Diameter of Aperture (mm)")
# plt.title("Aperture")
# plt.show()

mirror, length, width, distance = mirror(N, dx, theta, mirror_coord)


near = fresnel(aperture, mirror, lamda)
plt.xlabel("Length (m)")
plt.ylabel("Width (m)")
plt.title("Mirror Projection(Fresnel)")
plt.imshow(np.log(np.abs(near)), extent=[-mirror_coord[0], mirror_coord[0], -mirror_coord[1], mirror_coord[1]])
plt.show()
