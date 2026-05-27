import matplotlib.pyplot as plt
import numpy as np
import math

N = 100 #ideally(400)
radius = .25 #500 mm across!! (.25)

#window size created for dx adjustments (in far field ect)
window_size = (radius * 2)
dx = window_size/N
edge_taper = 0.01 #gaussian
theta =  45 #45 degrees
lamda = 3e-3 #3mm

#frame_size = [50,50]

#tilted mirror matrices
# phase_gradient = np.zeros((3,1))
# degree_tilt = np.array([[1,0,0], [0, math.cos(theta), math.sin(theta)], [0, -math.sin(theta), math.cos(theta)]])
# coord = np.array([[x], [y], [z]])

mirror_coord = np.array([2.7,1.8,4]) #m 
#but i might be having to use mm.....
# mirror_coord = np.array([2700, 1800, 4000]) #mm
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
def mirror_mask(mirror_data, mirror_type='flat',lamda=3e-3, mirror_tilt=False,theta=0):
    k = 2 * np.pi / lamda
    #2d coords from list
    x = mirror_data[0]
    y = mirror_data[1]

    if mirror_type == 'flat':
        #amplitude of 1 and phase change of 0
        base_mask = np.ones_like(x, dtype=complex)
    
    #different kinds of distortion
    elif mirror_type== 'distorted':
        pass
    else:
        raise ValueError("Unknown mirror type: {mirror_type}")
    if mirror_tilt:
        #turn from degree to radian
        radian = np.radians(theta)

        #calculate phase added by the tilt (Eulers formula?)
        tilt_phase = 2 * k * (x * np.tan(radian) + y * np.tan(radian))
        tilt_mask = np.exp(1j * tilt_phase)

        #apply tilt
        final_mask = base_mask * tilt_mask

    else:
        final_mask = base_mask

    return final_mask

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

            #multiply by pixel size

    return near_field   #as reflected ONTO mirror



def fraunhofer(a_grid, m_grid, lamda, dx, z1, near_grid):
    #Everything here just in meters i guessss
    #scaling difference for far field (should be the physical width of the source grid/ number of rows in )
    width_aperture = 1.8 #m
    rows_in_ap = a_grid.shape[0] #complex values
    new_dx = width_aperture / rows_in_ap

    #wave number
    k = 2 * np.pi / lamda
    #for the sake of not braking the code (propagation distance)
    zz = 200 #m (reccomended 2000 for Fresnel number of 0.12 which gives reasonable Fraunhofer approx)
    obs_sidelength = (lamda * zz) / new_dx
    obs_sample_interval = lamda * zz / width_aperture
    num_pixels = int(rows_in_ap)
    obs_coord =  np.linspace(-obs_sidelength/2, obs_sidelength/2 - obs_sample_interval, num_pixels) #start/stop
    [X2,Y2] = np.meshgrid(obs_coord,obs_coord)
    #complex amplitude scaling factor
    c = 1/(1j*lamda * zz) * np.exp(1j*k/(2*zz) * (X2 ** 2 + Y2 ** 2))
    #wave transformation 
    obs_plane = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(a_grid)))
    obs_plane_field = c * obs_plane * (new_dx ** 2)

    return obs_plane_field, X2, Y2

#     ########old function
 
#     #Reflection off of mirror
#     reflected = near_grid * m_grid

#     # FX, FY = np.meshgrid(fx, fy, indexing='ij')

#     # Fraunhofer (far field)
#     far = np.fft.fft2(np.fft.fftshift(reflected))

# #  # # Fresnel transfer function
# #     H = np.exp(-1j * np.pi * lamda * z1 * (FX**2 + FY**2))

#     center = np.fft.fftshift(far)

#     #intensity calc? !!!! Change to divided by peak and ln color
#     #center_I = center * (dx**2) / (1j * lamda * z1)

#     #linear intensity
#     max_intensity = np.abs(center)**2
#     #peak value
#     max = np.max(max_intensity)
#     index = np.unravel_index(np.argmax(max_intensity), max_intensity.shape)

#     #intensity normalized for graphing
#     normalized = max_intensity / max

#     # Frequency coordinates for fft (should be in radians but plot in degrees)
#     # fx = np.fft.fftfreq(N, d=dx/lamda)
#     # fy = np.fft.fftfreq(M, d=dx/lamda)
#     #!!!!! bug fix for indexing 100 rather than reading actual coordinates
#     rows, cols = max_intensity.shape
#     fx = np.fft.fftfreq(cols, d=new_dx)
#     fy = np.fft.fftfreq(rows, d=new_dx)
#     #Shift to center array alignment
#     fx_shifted = np.fft.fftshift(fx)
#     fy_shifted = np.fft.fftshift(fy)

#     #PLOTTING log intensity (max_intensity is linear intensity)
#     I_far = np.log10(normalized)

#     # #Real units (pixel locations into phsycial spatial frequencies)
#     x_far = fx_shifted[index[1]] #col 
#     y_far = fy_shifted[index[0]] #rows

#     #graph in degrees
#     x_theta = np.degrees(x_far)
#     y_theta = np.degrees(y_far)

#     # realigning peaks for physical spatial tracking in xy
#     # physical_x = x_theta[index[1]]
#     # physical_y = y_theta[index[0]]


#     return I_far, x_theta, y_theta



aperture, x, y, z = ap_grid(N, dx,radius, edge_taper, phase_gradient)

# PRINTING THE APERTURE
# plt.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
# plt.xlabel("Diameter of Aperture (mm)")
# plt.title("Aperture")
# plt.show()

mirror_data, length, width, distance = mirror(N, dx, theta, mirror_coord)

m_grid = mirror_mask(mirror_data, mirror_type='flat',lamda=3e-3, mirror_tilt=False,theta=0)

near = fresnel(aperture, x, y, mirror_data, lamda)  #might need to change mirror_data to m_grid and see what happens

# # PRINT NEAR FIELD MIRROR PROJECTION
# plt.xlabel("Length (m)")
# plt.ylabel("Width (m)")
# plt.title(f"Mirror Projection(Fresnel) Distance {distance} (m)")
# plt.imshow(np.log10(np.abs(near)**2), 
#            extent=[length.min(), length.max(), width.min(), width.max()], 
#            aspect='equal', 
#            origin='lower')
# plt.show()

I_far, x_far, y_far = fraunhofer(aperture, m_grid, lamda, dx, z1,near)

# #PRINTING FAR FIELD

# #perfectly center zoom?


# image boundaries!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
x_min, x_max = x_far.min(), x_far.max()
y_min, y_max = y_far.min(), y_far.max()

#inferno?
plt.imshow(np.log10(np.abs(I_far)**2), cmap='inferno', extent=[x_min, x_max, y_min, y_max], origin='lower')

# # Zoom
# plt.xlim(x_min / 10, x_max / 10)
# plt.ylim(y_min / 10, y_max / 10)

plt.colorbar(label='Intensity')
plt.title('Far-field diffraction pattern')
plt.show()



