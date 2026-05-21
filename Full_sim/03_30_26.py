import matplotlib.pyplot as plt
import numpy as np
import math

N = 100 #ideally(400)
radius = .25 #500 mm across!! (.25)
dx = (2 * radius)/N
edge_taper = 0.01 #gaussian
theta = math.pi / 4 #45 degrees (but in rad)
lamda = 3e-3 #3mm
#frame_size = [50,50]

#tilted mirror matrices
# phase_gradient = np.zeros((3,1))
# degree_tilt = np.array([[1,0,0], [0, math.cos(theta), math.sin(theta)], [0, -math.sin(theta), math.cos(theta)]])
# coord = np.array([[x], [y], [z]])

mirror_coord = np.array([2.7,1.8,4]) #m 
z1 = 4
# phase_gradient = (degree_tilt * coord) + mirror_coord
phase_gradient = 1

def ap_grid(N, dx,radius, edge_taper, phase_gradient):
    coords = (np.arange(N) - (N/2)) * dx
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

#Mirror non-flatness
def mirror_mask(mirror_data, mirror_type='flat',lamda, mirror_tilt=Fasle,theta):
    k = 2 * np.pi / lamda
    #2d coords from list
    x = mirror_data[0]
    y = mirror_data[1]

    if mirror_type == 'flat':
        #amplitude of 1 and phase change of 0
        return np.ones_like(x, dtype=complex)
    
    #different kinds of distortion
    elif mirror_type== 'distorted':
        pass
    else:
        raise ValueError("Unknown mirror type: {mirror_type}")
    if mirror_tilt:
        #turn from degree to radian
        radian = np.radians(theta)

        #calculate phase added by the tilt 
        tilt_phase = 2 * k * (x * np.tan(radian) + y * np.tan(radian))

def fresnel(aperture,x,y, mirror_data, lamda):
    #angular spatial frequency (optical wave number)
    k = 2 * np.pi / lamda
    dist = float(mirror_data[2])
    
    #from mirror (slicing for scaling)
    t_x = mirror_data[0][0, :]
    t_y = mirror_data[1][:, 0] 
    
    # (Rows Cols)
    near_field = np.zeros((len(t_y), len(t_x)), dtype=complex)
    
    for j in range(len(t_y)):     
        for i in range(len(t_x)): 
            d = np.sqrt((x - t_x[i])**2 + (y - t_y[j])**2 + dist**2)
            near_field[j, i] = np.sum(aperture * np.exp(1j * k * d) / d)

    return near_field   #as reflected ONTO mirror



def fraunhofer(a_grid, m_grid, lamda, dx, z1, near_grid):
    #freq coord
    #rows and cols
    N, M = a_grid.shape
    # k = 2 * np.pi / lamda
 
    #Reflection off of mirror
    reflected = near_grid * m_grid

    # FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Fraunhofer (far field)
    far = np.fft.fft2(np.fft.fftshift(reflected))
#  # # Fresnel transfer function
#     H = np.exp(-1j * np.pi * lamda * z1 * (FX**2 + FY**2))
    center = np.fft.fftshift(far)

    #intensity calc?
    center_I = center * (dx**2) / (1j * lamda * z1)

    I_far = np.abs(center_I)**2

    # Frequency coordinates for fft 
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dx)

    #Real units
    x_far = fx * lamda * z1
    y_far = fy * lamda * z1


    return I_far, x_far, y_far



aperture, x, y, z = ap_grid(N, dx,radius, edge_taper, phase_gradient)

#PRINTING THE APERTURE
# plt.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
# plt.xlabel("Diameter of Aperture (mm)")
# plt.title("Aperture")
# plt.show()

mirror_data, length, width, distance = mirror(N, dx, theta, mirror_coord)

m_grid = mirror_mask(mirror_data, mirror_type='flat',lamda, mirror_tilt=true,theta)

near = fresnel(aperture, x, y, mirror_data, lamda)  #might need to change mirror_data to m_grid and see what happens

#PRINT NEAR FIELD MIRROR PROJECTION
# plt.xlabel("Length (m)")
# plt.ylabel("Width (m)")
# plt.title(f"Mirror Projection(Fresnel) Distance {distance} (m)")
# plt.imshow(np.log10(np.abs(near)**2 + 1e-15), 
#            extent=[length.min(), length.max(), width.min(), width.max()], 
#            aspect='equal', 
#            origin='lower')
# plt.show()

I_far, x_far, y_far = fraunhofer(aperture, m_grid, lamda, dx, z1,near)

#PRINTING FAR FIELD

#perfectly center zoom?


# image boundaries!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
x_min, x_max = x_far.min(), x_far.max()
y_min, y_max = y_far.min(), y_far.max()

#inferno?
plt.imshow(I_far, cmap='inferno', extent=[x_min, x_max, y_min, y_max], origin='lower')

# Zoom
plt.xlim(x_min / 10, x_max / 10)
plt.ylim(y_min / 10, y_max / 10)
# plt.imshow(I_far, cmap='inferno')
plt.colorbar(label='Intensity')
plt.title('Far-field diffraction pattern')
plt.show()



