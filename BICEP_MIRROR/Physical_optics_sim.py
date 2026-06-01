import matplotlib.pyplot as plt
import numpy as np
import math


#can also abstract wavenumber to golabal variables


N = 100 #ideally(400)
radius = .25 #500 mm across!! (.25)

#window size created for dx adjustments (in far field ect)
window_size = (radius * 2)
dx = window_size/N
edge_taper = 0.01 #gaussian
theta_1 =  0 #45 degrees
theta_2 = 0
lamda = 3e-3 #3mm

#frame_size = [50,50]


mirror_coord = np.array([2.7,1.8,4]) #m 
#but i might be having to use mm.....
# mirror_coord = np.array([2700, 1800, 4000]) #mm

z1 = 200 #m (where thermal source is, but reccomended 2000 for Fresnel number of 0.12; which gives reasonable Fraunhofer approx)

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

    #Beam Steering
    k = 2 * np.pi / lamda 
    #parameters
    alpha_steer = np.radians(0) #(forwar/backwar tilt)
    theta_steer = np.radians(0) #roation of that tilt to steer in different quadrants

    tilt_phase = k * (x * np.cos(theta_steer) + y * np.sin(theta_steer)) * np.tan(alpha_steer)
    steered_aperture = aperture * np.exp(1j * tilt_phase)

    return steered_aperture, x, y, z #check output for complex output with xyz values


def mirror(N, dx, theta, mirror_coord):
    L_val = mirror_coord[0] / 2 
    W_val = mirror_coord[1] / 2 
    distance = mirror_coord[2]
    
    # Use N for the SHORTER side, and scale up for the LONGER side
    N_x = int(2 * L_val / dx)
    N_y = int(2 * W_val / dx)

    length_axis = (np.arange(N_x) - N_x//2) * dx ##Check dx here.....
    width_axis = (np.arange(N_y) - N_y//2) * dx

    x_grid, y_grid = np.meshgrid(length_axis, width_axis)


    #apply tilt coordinates forward along the x-axis
    X_tilt = x_grid
    Y_tilt = y_grid * np.cos(np.radians(theta))
    Z_tilt = distance + y_grid * np.sin(np.radians(theta))

    #distortion effects
    # height_of_bumps = 1 #peak height? 1m
    # length_of_bumps = 5 #pixels wide?

    # bumpiness = height_of_bumps * length_of_bumps

    # Z_tilt = z1 + bumpiness

    return [X_tilt, Y_tilt, Z_tilt], length_axis, width_axis, distance


def fresnel(aperture,x,y, mirror_data, lamda):
    #angular spatial frequency (optical wave number)
    k = 2 * np.pi / lamda
    dist = mirror_data[2]


    #from mirror (slicing for scaling)
    t_x_1d = mirror_data[0][0, :]
    t_y_1d = mirror_data[1][:, 0] 
    t_x, t_y = np.meshgrid(t_x_1d, t_y_1d) #puts into a 100 by 100 2d shape

    rows, cols = t_x.shape[0], t_x.shape[1]
    
    # (Rows Cols)
    near_field = np.zeros((rows, cols), dtype=complex)

    # d_rows, d_cols = dist.shape


    for j in range(rows):     
        for i in range(cols): 
            obs_x = t_x[j,i]
            obs_y = t_y[j,i]

            obs_dist = mirror_data[2][j,i]

            d = np.sqrt((x - obs_x)**2 + (y - obs_y)**2 + obs_dist**2)
            near_field[j, i] = np.sum(aperture * np.exp(1j * k * d) / d)

            #multiply by pixel size
    #phase adjustment
    # new_near_field = near_field * phase_tilt
    return near_field   #as reflected ONTO mirror



def fraunhofer(a_grid, lamda, window_size, dx, z1, near_grid):

    #if propagating from aperture vs if propagating from mirror
    width_aperture = window_size #m
    rows_in_ap = a_grid.shape[0] #complex values

    #scaling difference for far field (should be the physical width of the source grid/ number of rows in ) !!!!!! global variable? 
    new_dx = width_aperture / rows_in_ap

    #wave number
    k = 2 * np.pi / lamda

    #setting up variables for FFT far field (obs = observational)
    obs_sidelength = (lamda * z1) / new_dx
    obs_dx = lamda * z1 / width_aperture
    obs_coord =  np.linspace(-obs_sidelength/2, obs_sidelength/2 - obs_dx, int(rows_in_ap))
    [X2,Y2] = np.meshgrid(obs_coord,obs_coord)
    
    #complex amplitude scaling factor
    c = 1/(1j*lamda * z1) * np.exp(1j*k/(2*z1) * (X2 ** 2 + Y2 ** 2))

    #wave transformation 
    obs_plane = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(a_grid)))
    obs_plane_field = c * obs_plane * (new_dx ** 2)

    return obs_plane_field, X2, Y2


aperture, x, y, z = ap_grid(N, dx,radius, edge_taper, phase_gradient)

# PRINTING THE APERTURE
# plt.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
# plt.xlabel("Diameter of Aperture (mm)")
# plt.title("Aperture")
# plt.show()

mirror_data_1, length, width, distance = mirror(N, dx, theta_1, mirror_coord)

mirror_data_2, length, width, distance = mirror(N, dx, theta_2, mirror_coord)

# m_grid = mirror_mask(mirror_data, mirror_type='flat',lamda=3e-3, mirror_tilt=False,theta=0)


near = fresnel(aperture, x, y, mirror_data_1, lamda)  #might need to change mirror_data to m_grid and see what happens

near_2 = fresnel(aperture, x, y, mirror_data_2, lamda)

# PRINT NEAR FIELD MIRROR PROJECTION

#Subplots for mirror tilt
# fig, (one, two) = plt.subplots(1,2, figsize =( 14, 6))

# im1 = one.imshow(np.log10(np.abs(near)**2), 
#            extent=[length.min(), length.max(), width.min(), width.max()], 
#            aspect='equal', 
#            origin='lower')
# one.set_title(f"Mirror Projection {distance} (m) {theta_1} $^\circ$")
# one.set_xlabel("Length (m)")
# one.set_ylabel("Width (m)")

# im2 = two.imshow(np.log10(np.abs(near_2)**2), 
#            extent=[length.min(), length.max(), width.min(), width.max()], 
#            aspect='equal', 
#            origin='lower')
# two.set_title(f"Mirror Projection {distance} (m) {theta_2} $^\circ$")
# two.set_xlabel("Length (m)")
# two.set_ylabel("Width (m)")

# plt.show()

# #for single image
# plt.xlabel("Length (m)")
# plt.ylabel("Width (m)")
# plt.title(f"Mirror Projection(Fresnel) Distance {distance} (m)")
# plt.imshow(np.log10(np.abs(near)**2), 
#            extent=[length.min(), length.max(), width.min(), width.max()], 
#            aspect='equal', 
#            origin='lower')
# plt.show()

#PRINTING APERTURE LINEAR AND LOG

fig, (five, six) = plt.subplots(1,2, figsize = (14,6))

img5 = five.imshow(np.abs(near)**2, extent=[length.min(), length.max(), width.min(), width.max()], 
          aspect='equal', 
           origin='lower')
five.set_title(f"Mirror Projection Linear")
fig.colorbar(img5, ax=five, label='Intensity')

img6 = six.imshow(np.log10(np.abs(near)**2),extent=[length.min(), length.max(), width.min(), width.max()], 
          aspect='equal', 
           origin='lower')
six.set_title(f"Mirror Projection Log")
fig.colorbar(img6, ax=six, label='Intensity')

plt.show()

# I_far, x_far, y_far = fraunhofer(aperture, lamda, window_size, dx, z1,near)

# # #PRINTING FAR FIELD 

# x_min, x_max = x_far.min(), x_far.max()
# y_min, y_max = y_far.min(), y_far.max()

#single image
# plt.imshow(np.log10(np.abs(I_far)**2), extent=[x_min, x_max, y_min, y_max], origin='lower')
# plt.colorbar(label='Intensity')
# plt.title('Far-field diffraction pattern')
# plt.show()

#subplots (log and linear sidebyside)

# fig, (three, four) = plt.subplots(1,2, figsize = (14,6))

# img3 = three.imshow(np.abs(I_far)**2,extent=[x_min, x_max, y_min, y_max], origin='lower')
# three.set_title(f"Far-Field diffraction linear")
# fig.colorbar(img3, ax=three, label='Intensity')

# img4 = four.imshow(np.log10(np.abs(I_far)**2), extent=[x_min, x_max, y_min, y_max], origin='lower')
# four.set_title(f"Far-Field diffraction Log")
# fig.colorbar(img4, ax=four, label='Intensity')

# plt.show()




