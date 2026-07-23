import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from skimage.registration import phase_cross_correlation

#can also abstract wavenumber to golabal variables

#aperture
#change to [nXm] resolution
N = 100 #ideally(400)
#[] dimensions
radius = .25 #500 mm across!! (.25) aperture

#mirror

#window size created for dx adjustments (in far field ect)
window_size = (radius * 2)
dx = window_size/N
edge_taper = 0.01 #gaussian
theta_1 =  45 #45 degrees
lamda = 2e-3 #3mm
#mainly for side by side comparisons
theta_2 = 0


#frame_size = [50,50]


mirror_coord = np.array([2.7,1.8,4]) #m 
#but i might be having to use mm.....
# mirror_coord = np.array([2700, 1800, 4000]) #mm

z1 = 200 #m (where thermal source is, but reccomended 2000 for Fresnel number of 0.12; which gives reasonable Fraunhofer approx)

# phase_gradient = (degree_tilt * coord) + mirror_coord
#not sure this is needed atm
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


def mirror(N, dx, theta, mirror_coord,lamda):
    k = 2* np.pi / lamda

    L_val = mirror_coord[0] / 2 #to center
    W_val = mirror_coord[1] / 2 
    distance = mirror_coord[2]
    
    # Use N for the SHORTER side, and scale up for the LONGER side
    N_x = int(2 * L_val / dx)
    N_y = int(2 * W_val / dx)

    length_axis = (np.arange(N_x) - N_x//2) * dx ##Check dx here.....dividing by dx then multiplying
    width_axis = (np.arange(N_y) - N_y//2) * dx

    x_grid, y_grid = np.meshgrid(length_axis, width_axis)


    #apply tilt coordinates forward along the x-axis
    X_tilt = x_grid
    Y_tilt = y_grid * np.cos(np.radians(theta))
    Z_tilt = distance + y_grid * np.sin(np.radians(theta))

    #phase adjustment for mirror reflection at an angle (possible error)
    # phase_tilt_adjustment = np.exp(1j*Y_tilt*k)

    # x_reflect = X_tilt * phase_tilt_adjustment
    # y_reflect = Y_tilt * phase_tilt_adjustment
    # z_reflect = Z_tilt * phase_tilt_adjustment

    return [X_tilt, Y_tilt, Z_tilt], length_axis, width_axis, distance

def load_points(filename):
    df = pd.read_csv(filename, sep='\t', header=2)
    df = pd.read_csv(filename, header=2)
    df = df.rename(columns={"X (project units)": "X", "Y (project units)": "Y", "Z (project units)": "Z (um)", "Z Precision": "Z Precision (um)"})
    df = df[df.Id>12]
    df[['Z (um)','Z Precision (um)']] = df[['Z (um)','Z Precision (um)']]*1000000
    return df

def bilinear_interpolation(mirror_data_1):
    #import data points
    Cold = load_points(r"C:\Users\lj350\Downloads\BICEP\BICEP_MIRROR\cold.txt")

    space = Cold[["X", "Y"]].to_numpy()
    depth = Cold["Z (um)"].to_numpy() #change to mm soon!!!!!!

    #coordinate offset shift (because the photo data is defined at different points than the new graph. Zero Zero shifted)
    x_center_error = 0.1   # (-1.4 + 1.6) / 2
    y_center_error = 0.25  # (-0.75 + 1.25) / 2

    #subtract to the true (0,0) center
    space[:, 0] -= x_center_error
    space[:, 1] -= y_center_error

    #interpolate data points 
    linear_interpolation = LinearNDInterpolator(space,depth)

    #assign to mesh grid 
    mirror_interpolation = np.c_[mirror_data_1[0].ravel(), mirror_data_1[1].ravel()]
    mirror_new = linear_interpolation(mirror_interpolation)

    #extrapolation
    # nearest_interpolation = NearestNDInterpolator(space, depth)
    # mirror_new_nearest = nearest_interpolation(mirror_interpolation)

    #testing smooth extrapolation
    # extrapolation (Calculates the smooth slopes for the background)
    outside_interp = RBFInterpolator(space, depth, kernel='linear')
    mirror_new_rbf = outside_interp(mirror_interpolation)

    #fill the cut-off with nearest data (but causes binning....)
    # mirror_new = np.where(np.isnan(mirror_new_nearest), mirror_new_nearest, mirror_new_nearest)
    # fill the cut-off smoothly (No more binning!)
    mirror_new = np.where(np.isnan(mirror_new), mirror_new_rbf, mirror_new)


    #return new interpolated data points to compare to flat mirror ect
    mirror_mesh_z = mirror_new.reshape(mirror_data_1[0].shape)

    return [mirror_data_1[0], mirror_data_1[1], mirror_mesh_z]

#slightly edited from sourced code(used for side by side comparison)
def plot_points(df, meas, min, max, mirror_data, mirrordata,new_mirror_z):
    fig, (one,two) = plt.subplots(1,2, figsize=(14,6))

    im1 = one.pcolormesh(mirror_data,mirrordata,new_mirror_z,vmin=min, vmax=max)
    fig.colorbar(im1, ax=one, label=meas)

    one.set_title("Mirror Linear Interpolation and Extrapolation")
    one.set_xlabel("X (m)")
    one.set_ylabel("Y (m)")

    one.grid(True)

    one.set_aspect('equal', adjustable='box')

    #scatter plot
    im2 = two.scatter(x=df['X'], y=df['Y'], c=df[meas], s=200, vmin=min, vmax=max)
    two.set_aspect('equal')
    two.grid()
    two.set_title("Photogrammetry Data")
    two.set_xlabel('x (m)')
    two.set_ylabel('y (m)')
    two.set_xlim(-1.4, 1.6)
    fig.colorbar(im2, ax=two, label=meas)
    
    plt.show()

def fresnel(aperture,x,y, mirror_data, lamda):
    #angular spatial frequency (optical wave number)
    k = 2 * np.pi / lamda

    #from mirror (slicing for scaling)
    t_x_1d = mirror_data[0][0, :]
    t_y_1d = mirror_data[1][:, 0] 
    t_x, t_y = np.meshgrid(t_x_1d, t_y_1d) #puts into a 100 by 100 2d shape

    rows, cols = t_x.shape[0], t_x.shape[1]
    
    # (Rows Cols)
    near_field = np.zeros((rows, cols), dtype=complex)


    for j in range(rows):     
        for i in range(cols): 
            obs_x = t_x[j,i]
            obs_y = t_y[j,i]

            obs_dist = mirror_data[2][j,i]

            d = np.sqrt((x - obs_x)**2 + (y - obs_y)**2 + obs_dist**2)
            near_field[j, i] = (np.sum(aperture * np.exp(1j * k * d) / d))

            #for complex number multiplication to occur
            # phase_tilt_adjustment = np.exp(1j*obs_y*k)
            # point = near_field[j, i]
            # #adjustment for mirror reflection
            # near_field[j, i] = point * phase_tilt_adjustment
            
    return near_field   #as reflected ONTO mirror
    
def fraunhofer(lamda, window_size, z1, near_grid, mirror_coord):
    k = 2 * np.pi / lamda


    rows_in_mirror = near_grid.shape[0]
    col_mirror = near_grid.shape[1]
    dx = mirror_coord[0] / col_mirror
    dy = mirror_coord[1] / rows_in_mirror
    
    #create mesh grid for the mirror in this function (centered at 00)
    y_mir_index = np.arange(rows_in_mirror) - (rows_in_mirror //2)
    x_mir_index = np.arange(col_mirror) - (col_mirror //2)

    X_mirror, Y_mirror = np.meshgrid(x_mir_index, y_mir_index)

    #45 shift for unpadded (????)
    z_small = Y_mirror * np.sin(np.pi / 4) + 4
    y_small = Y_mirror * np.cos(np.pi / 4) + 0
    x_small = X_mirror + 0

    #set center for mirror
    
    padded_center = 512

    #starting spots for each side of the mirror. shape (360,540)
    row_start = padded_center - (360 //2)
    col_start = padded_center - (540//2)

    #creating the mirror padding
    npad = [1024, 1024] 
    f2 = np.zeros(shape=(npad[0], npad[1]), dtype=complex)
    # #to center (accounting for offset)
    # ix = (f2.shape[0] - near_grid.shape[0]) // 2  # Row offset
    # iy = (f2.shape[1] - near_grid.shape[1]) // 2  # Column offset
    
    # row and col dim (!!!!!might switch iy and ix)
    f2[row_start : row_start + 360, col_start : col_start + 540] = near_grid

    near_grid = f2

      # #setting up variables for FFT far field (obs = observational)
    # #spatial freq for x and y 
    fx = np.fft.fftshift(np.fft.fftfreq(near_grid.shape[1], d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(near_grid.shape[0], d=dy))
    X2, Y2 = np.meshgrid(fx, fy)

    #convert to cos
    alpha = X2 * lamda
    beta = Y2 * lamda

    #for plotting fft in degrees from center
    x_deg = np.arcsin(alpha) * (180/ np.pi)
    y_deg = np.arcsin(beta) * (180/np.pi)

    obs_plane = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(near_grid)))

    fraunhoffer_phase_factor = np.exp(1j * k * y_small)

    #you do have to 'double pad' but the second padding is for the fraunhoffer phase factor;
    shape = (1024,1024)
    shape_row = shape[0] - fraunhoffer_phase_factor.shape[0]
    shape_col = shape[1] - fraunhoffer_phase_factor.shape[1]

    pad_top = shape_row //2
    pad_bottom = shape_row - pad_top
    pad_left = shape_col // 2 
    pad_right = shape_col - pad_left

    padded_phase_factor = np.pad(
    fraunhoffer_phase_factor, 
    ((pad_top, pad_bottom), (pad_left, pad_right)))

    tilted_near_grid = obs_plane * padded_phase_factor

    uv_width = [np.degrees(0.5 * lamda / (X2[0,1] - X2[0,0])), np.degrees(0.5 * lamda / (Y2[1,0] - Y2[0,0]))]

    return tilted_near_grid, uv_width, x_deg, y_deg


aperture, x, y, z = ap_grid(N, dx,radius, edge_taper, phase_gradient)

mirror_data_1, length, width, distance = mirror(N, dx, theta_1, mirror_coord,lamda)


# new_mirror_z = bilinear_interpolation(mirror_data_1)

# # mirror plotting
# Set_Cold = load_points(r"C:\Users\lj350\Downloads\BICEP\BICEP_MIRROR\cold.txt")
# plot_points(Set_Cold, "Z (um)", -400, 1500, mirror_data_1[0],mirror_data_1[1],new_mirror_z[2])

#flat mirror
near = fresnel(aperture, x, y, mirror_data_1, lamda)

# #for altered mirror states
# near = fresnel(aperture, x, y , new_mirror_z, lamda)

I_far, uv_width, x_degree, y_degree = fraunhofer(lamda, window_size, z1,near, mirror_coord)

x_min, x_max = x_degree.min(), x_degree.max()
y_min, y_max = y_degree.min(), y_degree.max()


#new printing layout six side by side
fig, ((ap, close, far), (ap_phase, close_phase , far_phase)) = plt.subplots(2, 3, figsize=(8,12))

ap1 = ap.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
ap.set_title("Aperture Amplitude")
ap.set_ylabel("Diameter of Aperture (m)")

close1 = close.imshow(np.log10(np.abs(near)**2), 
            extent=[length.min(), length.max(), width.min(), width.max()], 
            aspect='equal', 
            origin='lower')
close.set_title("Mirror Projection Amplitude")
close.set_xlabel("Length (m)")
close.set_ylabel("Width (m)")

far1 = far.imshow(np.log10(np.abs(I_far)**2) , extent=[x_min, x_max, y_min, y_max])
far.set_title("Far-field Amplitude")
fig.colorbar(far1, ax=far, label="Intensity")
far.set_xlabel("Degrees")
far.set_ylabel("Degrees")
far.set_xlim(-3,3)
far.set_ylim(-3,3)

ap2 = ap_phase.imshow(np.atan2(np.imag(aperture), np.real(aperture)), extent = [-radius, radius, -radius, radius])
ap_phase.set_title("Aperture Phase")
ap_phase.set_ylabel("Diameter of Aperture (m)")

close2 = close_phase.imshow(np.atan2(np.imag(near), np.real(near)), 
            extent=[length.min(), length.max(), width.min(), width.max()], 
            aspect='equal', 
            origin='lower')
close_phase.set_title("Mirror Projection Phase")
close_phase.set_xlabel("Length (m)")
close_phase.set_ylabel("Width (m)")

far2 = far_phase.imshow(np.atan2(np.imag(I_far), np.real(I_far)), extent=[x_min, x_max, y_min, y_max], origin='lower')
far_phase.set_title("Far-field Phase")
fig.colorbar(far2, ax=far_phase, label="Intensity")
far_phase.set_xlabel("Degrees")
far_phase.set_ylabel("Degrees")
far_phase.set_xlim(-3,3)
far_phase.set_ylim(-3,3)

plt.tight_layout()
plt.show()

