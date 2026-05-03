import matplotlib.pyplot as plt
import numpy as np
import math

N = 400 #ideally(400)
radius = .25 #500 mm across!! (.25)
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
mirror_coord = np.array([1.5,1.5,4]) #m (6,4,4)

# phase_gradient = (degree_tilt * coord) + mirror_coord
phase_gradient = 1

def ap_grid(N, dx,radius, edge_taper, phase_gradient):
    coords = (np.arange(N) - N/2) * dx
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
    L_val = mirror_coord[0] / 2 
    W_val = mirror_coord[1] / 2 
    distance = mirror_coord[2]
    
    # Use N for the SHORTER side, and scale up for the LONGER side
    N_x = int(2 * L_val / dx)
    N_y = int(2 * W_val / dx)

    length_axis = (np.arange(N_x) - N_x//2) * dx
    width_axis = (np.arange(N_y) - N_y//2) * dx

    x_grid, y_grid = np.meshgrid(length_axis, width_axis)
    return [x_grid, y_grid, distance], length_axis, width_axis, distance





def fresnel(aperture,x,y, mirror_data, lamda):
    k = 2 * np.pi / lamda
    dist = float(mirror_data[2])
    
    #from mirror
    t_x = mirror_data[0][0, :]
    t_y = mirror_data[1][:, 0] 
    
    # (Rows Cols)
    near_field = np.zeros((len(t_y), len(t_x)), dtype=complex)
    
    for j in range(len(t_y)):     
        for i in range(len(t_x)): 
            d = np.sqrt((x - t_x[i])**2 + (y - t_y[j])**2 + dist**2)
            near_field[j, i] = np.sum(aperture * np.exp(1j * k * d) / d)

    return near_field






#def fraunhofer(a_grid, m_grid, lamda):


aperture, x, y, z = ap_grid(N, dx,radius, edge_taper, phase_gradient)

pad = 0
aperture_padded = np.pad(aperture, ((pad, pad), (pad, pad)), mode='constant')

N_new = aperture_padded.shape[0]
coords = np.linspace(-radius, radius, N)
x_padded, y_padded = np.meshgrid(coords, coords)

# plt.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
# plt.xlabel("Diameter of Aperture (mm)")
# plt.title("Aperture")
# plt.show()

mirror_data, length, width, distance = mirror(N, dx, theta, mirror_coord)


near = fresnel(aperture_padded, x_padded, y_padded, mirror_data, lamda)
plt.xlabel("Length (m)")
plt.ylabel("Width (m)")
plt.title(f"Mirror Projection(Fresnel) Distance {distance} (m)")
plt.imshow(np.log10(np.abs(near)**2 + 1e-15), 
           extent=[length.min(), length.max(), width.min(), width.max()], 
           aspect='equal', 
           origin='lower')


plt.show()
