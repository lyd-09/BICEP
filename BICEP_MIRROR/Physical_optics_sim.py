import matplotlib.pyplot as plt
import numpy as np
import time 
from tqdm import tqdm
import math
from numba import njit, prange
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator, CloughTocher2DInterpolator
from skimage.registration import phase_cross_correlation

#can also abstract wavenumber to golabal variables

#aperture
#change to [nXm] resolution
N = 125 #ideally(400)nope ideally (125)
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


mirror_coord = np.array([2.7,1.8,4]) #m 
#but i might be having to use mm.....
# mirror_coord = np.array([2700, 1800, 4000]) #mm

z1 = 200 #m (where thermal source is, but reccomended 2000 for Fresnel number of 0.12; which gives reasonable Fraunhofer approx)

# phase_gradient = (degree_tilt * coord) + mirror_coord
#not sure this is needed atm
phase_gradient = 1
k = 2 * np.pi / lamda 

def ap_grid(N, dx, k, radius, edge_taper, phase_gradient):
    coords = (np.arange(N) - (N/2)) * dx
    x, y = np.meshgrid(coords, coords)

    # aperture calc
    r = np.sqrt(x**2 + y**2)
    inside = r < radius

    # calc 2d edge taper
    alpha = -np.log(edge_taper) / radius**2
    amplitude_profile = np.exp(-alpha * r**2)

    # beam steering across whole grid
    theta_steer = np.radians(0) # rotation angle into different quadrants
    alpha_steer = np.radians(0) # forward/backward tilt

    # tilt phase calculation for whole grid
    tilt_phase = k * (x * np.cos(theta_steer) + y * np.sin(theta_steer)) * np.tan(alpha_steer)
    adjustment = np.exp(1j * tilt_phase)

    full_steered_field = amplitude_profile * adjustment

    # aperture boundary
    aperture = np.zeros(shape=(N,N), dtype=complex)
    aperture[inside] = full_steered_field[inside]


    return aperture, x, y

def mirror(dx, theta, mirror_coord):

    L_val = mirror_coord[0] / 2 #to center
    W_val = mirror_coord[1] / 2 
    distance = mirror_coord[2]
    
    # Use N for the SHORTER side, and scale up for the LONGER side
    N_x = int(2 * L_val / dx)
    N_y = int(2 * W_val / dx)

    length_axis = (np.arange(N_x) - N_x//2) * dx ##Check dx here.....dividing by dx then multiplying
    width_axis = (np.arange(N_y) - N_y//2) * dx
    #for when pixel size is increased
    # length_axis = np.linspace(-L_val, L_val, N_x)
    # width_axis = np.linspace(-W_val, W_val, N_y)

    x_grid, y_grid = np.meshgrid(length_axis, width_axis)

    #apply tilt coordinates forward along the y-axis
    X_tilt = x_grid
    Y_tilt = y_grid * np.cos(np.radians(theta))
    Z_tilt = distance + y_grid * np.sin(np.radians(theta))

    return [X_tilt, Y_tilt, Z_tilt], length_axis, width_axis, distance

def load_points(filename):
    df = pd.read_csv(filename, sep='\t', header=2)
    df = pd.read_csv(filename, header=2)
    df = df.rename(columns={"X (project units)": "X", "Y (project units)": "Y", "Z (project units)": "Z (um)", "Z Precision": "Z Precision (um)"})
    df = df[df.Id>12]
    # df[['Z (um)','Z Precision (um)']] = df[['Z (um)','Z Precision (um)']]*1000000
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

    # extrapolation (Calculates the smooth slopes for the background)
    outside_interp = RBFInterpolator(space, depth, kernel='linear')
    mirror_new_rbf = outside_interp(mirror_interpolation)

    # fill the cut-off smoothly
    mirror_new = np.where(np.isnan(mirror_new), mirror_new_rbf, mirror_new)


    #return new interpolated data points to compare to flat mirror ect
    mirror_mesh_z = mirror_new.reshape(mirror_data_1[0].shape)

    total_mirror_z_meters =  mirror_mesh_z

    return [mirror_data_1[0], mirror_data_1[1], total_mirror_z_meters]

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

def fresnel(aperture,x,y, mirror_data, k):
    rows, cols = mirror_data[0].shape[0], mirror_data[0].shape[1]

    # (Rows Cols)
    near_field = np.zeros((rows, cols), dtype=np.complex128)

    for j in tqdm(range(rows)):     
        for i in range(cols): 
            obs_x = mirror_data[0][j,i]
            obs_y = mirror_data[1][j,i]
            obs_dist = mirror_data[2][j,i]

            d = np.sqrt((x - obs_x)**2 + (y - obs_y)**2 + obs_dist**2)
            near_field[j, i] = (np.sum(aperture * np.exp(1j * k * d) / d))
      
    return near_field   #as reflected ONTO mirror

def apply_phase_array_nonflat(wavenumber, near, new_mirror_z):
    phi = wavenumber * 2 * new_mirror_z[2]

    #creating a complex phase mask
    phase_mask = np.exp(1j * phi)

    near_nonflat = near * phase_mask

    return near_nonflat


def fraunhofer(theta, k, near_grid, mirror_coord):

    rows_in_mirror = near_grid.shape[0]
    col_mirror = near_grid.shape[1]
    dx = mirror_coord[0] / col_mirror
    dy = mirror_coord[1] / rows_in_mirror

# mesh of near_grid (360, 540) # Ny = 360, Nx = 540
    Ny, Nx = near_grid.shape
    y_vec = np.arange(-Ny//2, Ny//2) * dy
    x_vec = np.arange(-Nx//2, Nx//2) * dx
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)

    # demodulate using the Y of the near field
    fraunhoffer_phase_factor = np.exp(-1j * k * Y_mesh * np.sin(np.radians(theta)))
    demodulated_grid = near_grid * fraunhoffer_phase_factor

    #padding
    npad = [1024, 1024] 
    f2 = np.zeros(shape=(npad[0], npad[1]), dtype=complex)

    padded_center = 512
    row_start = padded_center - (Ny // 2)
    col_start = padded_center - (Nx // 2)

    f2[row_start : row_start + Ny, col_start : col_start + Nx] = demodulated_grid

    obs_plane = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f2)))

    fx = np.fft.fftshift(np.fft.fftfreq(npad[1], d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(npad[0], d=dy))
    X2, Y2 = np.meshgrid(fx, fy)

    x_limit_deg = np.degrees(0.5 * lamda / dx)
    y_limit_deg = np.degrees(0.5 * lamda / dy)

    return obs_plane, x_limit_deg, y_limit_deg


aperture, x, y= ap_grid(N, dx, k, radius, edge_taper, phase_gradient)

mirror_data_1, length, width, distance = mirror(dx, theta_1, mirror_coord)

#for new phase output of non flat mirror
new_mirror_z = bilinear_interpolation(mirror_data_1)


# # mirror plotting
# Set_Cold = load_points(r"C:\Users\lj350\Downloads\BICEP\BICEP_MIRROR\cold.txt")
# plot_points(Set_Cold, "Z (um)", -400, 1500, mirror_data_1[0],mirror_data_1[1],new_mirror_z[2])

#prior to running vectorized approach



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#flat mirror
near = fresnel(aperture, x, y, mirror_data_1, k)

non_flat = apply_phase_array_nonflat(k, near, new_mirror_z)
# #for altered mirror states
# near = fresnel(aperture, x, y , new_mirror_z, lamda)

I_far, x_degree, y_degree = fraunhofer(theta_1, k, non_flat, mirror_coord)

I_far_original, x_degree, y_degree = fraunhofer(theta_1, k, near, mirror_coord)

#for non log plot 
difference_fft = (I_far_original - I_far).real

#unofficial y axis adjustment 
x_min, x_max = -x_degree, x_degree

if theta_1 == 45:

    y_min, y_max = -y_degree * (2 ** 0.5), y_degree * (2 ** 0.5)

else:

    y_min, y_max = -y_degree, y_degree


#plotting the difference between far fields

fig, ((orig, non, diff), (orig_phase, non_phase, diff_phase)) = plt.subplots(2,3, figsize=(8,12))

orig1 = orig.imshow(np.log10(np.abs(I_far_original)**2), extent=[x_min, x_max, y_min, y_max])
orig.set_title("Flat mirror FFT")
orig.set_xlim(-3,3)
orig.set_ylim(-3,3)

non1 = non.imshow(np.log10(np.abs(I_far)**2), extent=[x_min, x_max, y_min, y_max])
non.set_title("Non-flat mirror FFT")
non.set_xlim(-3,3)
non.set_ylim(-3,3)

diff1 = diff.imshow(difference_fft, extent=[x_min, x_max, y_min, y_max])
diff.set_title("Difference")
diff.set_xlim(-3,3)
diff.set_ylim(-3,3)

orig_phase1 = orig_phase.imshow(np.atan2(np.imag(I_far_original), np.real(I_far_original)), extent=[x_min, x_max, y_min, y_max])
orig_phase.set_title("Original Phase")
orig_phase.set_xlim(-3,3)
orig_phase.set_ylim(-3,3)

non_phase1 = non_phase.imshow(np.atan2(np.imag(I_far), np.real(I_far)),extent=[x_min, x_max, y_min, y_max])
non_phase.set_title("Non-flat phase")
non_phase.set_xlim(-3,3)
non_phase.set_ylim(-3,3)

diff_phase1 = diff_phase.imshow(np.atan2(np.imag(difference_fft), np.real(difference_fft)), extent=[x_min, x_max, y_min, y_max])
diff_phase.set_title("Difference phase")
diff_phase.set_xlim(-3,3)
diff_phase.set_ylim(-3,3)

plt.tight_layout()
plt.show()



# #new printing layout six side by side
# fig, ((ap, close, far), (ap_phase, close_phase , far_phase)) = plt.subplots(2, 3, figsize=(8,12))

# ap1 = ap.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
# ap.set_title("Aperture Amplitude")
# ap.set_ylabel("Diameter of Aperture (m)")

# close1 = close.imshow(10 * (np.log10(np.abs(near)**2) - np.log10(np.abs(near).max()**2)), 
#             extent=[length.min(), length.max(), width.min(), width.max()], 
#             aspect='equal', 
#             origin='lower')
# close.set_title("Mirror Projection Amplitude")
# close.set_xlabel("Length (m)")
# close.set_ylabel("Width (m)")

# far1 = far.imshow((np.log10(np.abs(I_far)**2)) , extent=[x_min, x_max, y_min, y_max])
# far.set_title("Far-field Amplitude")
# fig.colorbar(far1, ax=far, label="Intensity")
# far.set_xlabel("Degrees")
# far.set_ylabel("Degrees")
# # far.set_xlim(-3,3)
# # far.set_ylim(-3,3)

# ap2 = ap_phase.imshow(np.atan2(np.imag(aperture), np.real(aperture)), extent = [-radius, radius, -radius, radius])
# ap_phase.set_title("Aperture Phase")
# ap_phase.set_ylabel("Diameter of Aperture (m)")

# close2 = close_phase.imshow(np.atan2(np.imag(near), np.real(near)), 
#             extent=[length.min(), length.max(), width.min(), width.max()], 
#             aspect='equal', 
#             origin='lower')
# close_phase.set_title("Mirror Projection Phase")
# close_phase.set_xlabel("Length (m)")
# close_phase.set_ylabel("Width (m)")

# far2 = far_phase.imshow(np.atan2(np.imag(I_far), np.real(I_far)), extent=[x_min, x_max, y_min, y_max], origin='lower')
# far_phase.set_title("Far-field Phase")
# fig.colorbar(far2, ax=far_phase, label="Intensity")
# far_phase.set_xlabel("Degrees")
# far_phase.set_ylabel("Degrees")
# # far_phase.set_xlim(-3,3)
# # far_phase.set_ylim(-3,3)

# plt.tight_layout()
# plt.show()
