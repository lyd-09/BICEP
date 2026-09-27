import matplotlib.pyplot as plt
import numpy as np
import os
import time 
from tqdm import tqdm
import math
from numba import njit, prange
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator, CloughTocher2DInterpolator
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
import datetime

#can also abstract wavenumber to golabal variables
#for unique save inputs
date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#aperture
#change to [nXm] resolution
N = 125 #ideally(400)nope ideally (125)
#padding dimensions for mirror before fft: 1024, 2048,4096
N_mirror = 2048
radius = .25 #500 mm across!! (.25) aperture

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

z_thermal = 200 #m (where thermal source is, but reccomended 2000 for Fresnel number of 0.12; which gives reasonable Fraunhofer approx)

# phase_gradient = (degree_tilt * coord) + mirror_coord
#not sure this is needed atm
phase_gradient = 1
k = 2 * np.pi / lamda 

def ap_grid(N, dx, N_mirror, k, radius, edge_taper, phase_gradient):
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


def fraunhofer(N_mirror,theta, k, near_grid, mirror_coord):

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

    #padding for finer res of fft
    Ny_pad = N_mirror
    Nx_pad = int(N_mirror * (Nx / Ny)) #spacing for rectangular mirror (1.5 step scaling)

    f2 = np.zeros((Ny_pad, Nx_pad), dtype=complex)

    row_start = (Ny_pad //2) - (Ny // 2)
    col_start = (Nx_pad //2) - (Nx //2)

    f2[row_start : row_start + Ny, col_start : col_start + Nx] = demodulated_grid

    obs_plane = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f2)))
    
    fx = np.fft.fftshift(np.fft.fftfreq(Nx_pad, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny_pad, d=dy))
    fy_projected = fy / np.cos(np.radians(theta))
    # X2, Y2 = np.meshgrid(fx, fy)
    X2, Y2 = np.meshgrid(fx, fy_projected)

    x_limit_deg = np.degrees(0.5 * lamda / dx)
    y_limit_deg = np.degrees(0.5 * lamda / dy)

    #for the plotting of differences and stuff
    norm = np.abs(obs_plane) ** 2

    #helps with ensuring when the padding is changed the physical spacing is too

    dfx = 1 / (Nx_pad * dx)
    
    dfy = 1/ (Ny_pad * dy)

    #integral normalization
    total_integral = np.sum(norm) * dfx * dfy

    normalized_integral = norm / total_integral 

    #peak normlz
    normalized_peak = norm / np.max(norm)

    return obs_plane, normalized_integral, normalized_peak, x_limit_deg, y_limit_deg, X2, Y2

def gaussian_fit(M, omega, mu_x, mu_y, sigma_x, sigma_y, theta):
    x, y = M
    dx = x - mu_x
    dy = y - mu_y

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # rotated 2D Gaussian coefficients
    a = (cos_t**2 / (2 * sigma_x**2)) + (sin_t**2 / (2 * sigma_y**2))
    b = (-np.sin(2 * theta) / (4 * sigma_x**2)) + (np.sin(2 * theta) / (4 * sigma_y**2))
    c = (sin_t**2 / (2 * sigma_x**2)) + (cos_t**2 / (2 * sigma_y**2))

    exponent = -(a * dx**2 + 2 * b * dx * dy + c * dy**2)
    
    intensity_difference = omega * np.exp(exponent)

    return intensity_difference


def plot_2d_gaussian(type_of_mirror_analysis,initial_thing_to_plot, X2, Y2, fit_map_2d, residual_map, popt, save_path):
    #plotting 2d gaussian fit to the normalized integral difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #extracting covariance elements
    optimal_xx = popt[3]
    optimal_yy = popt[4]
    optimal_xy = popt[5]

    #get covariance matrix and eigenvalues
    cov_matrix = np.array([[optimal_xx, optimal_xy],
                           [optimal_xy, optimal_yy]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    #unwarped beam widths
    true_x2 = eigenvalues[0]
    true_y2 = eigenvalues[1]

    dominant_vector = eigenvectors[:,1]
    tilt_angle_degree = np.degrees(np.arctan2(dominant_vector[1],dominant_vector[0]))
    #saving 6 data parameters
    with open(data_name, "w") as file:
        file.write(f"Peak amplitude (1/omega): {1.0 / optimized_omega:.5f}\n")
        file.write(f"Center shift: (X: {opt_mu_x:.4f}, Y: {opt_mu_y:.4f})\n")
        file.write(f"True physical widths (Tilt-corrected): \n")
        file.write(f"r_x^2 = {true_x2:.2f}, r_y^2 = {true_y2:.4f}\n")
        file.write(f"Beam tilt angle: {tilt_angle_degree:.2f}\n")
        file.write(f"Covariance parameters: xx = {optimal_xx:.4f}, yy = {optimal_yy:.4f}, xy = {optimal_xy:.4f}\n")

    print(f"Data saved to: {data_name}")

    #uniform color scale
    v_min = np.min(initial_thing_to_plot)
    v_max = np.max(initial_thing_to_plot)
    zoom_factor = 5.0  

    # Center the zoom box on your optimized center shifts
    x_center = opt_mu_x
    y_center = opt_mu_y

    # Set the new bounding box dimensions based on the actual beam sizes
    x_min = x_center - (zoom_factor * opt_sigma_x)
    x_max = x_center + (zoom_factor * opt_sigma_x)

    y_min = y_center - (zoom_factor * opt_sigma_y)
    y_max = y_center + (zoom_factor * opt_sigma_y)

    #raw sim data
    im1 = axes[0].pcolormesh(X2, Y2, initial_thing_to_plot, vmin=v_min, vmax=v_max, shading='auto')
    axes[0].set_title(type_of_mirror_analysis)
    axes[0].set_xlabel("Frequency Coordinates")
    fig.colorbar(im1, ax=axes[0])
    axes[0].set_xlim(x_min,x_max)
    axes[0].set_ylim(y_min,y_max)

    #gaussian
    im2 = axes[1].pcolormesh(X2, Y2, fit_map_2d, shading='auto')
    axes[1].set_title("Gaussian Fit")
    axes[1].set_xlabel("Frequency Coordinates")
    fig.colorbar(im2, ax=axes[1])
    axes[1].set_xlim(x_min,x_max) 
    axes[1].set_ylim(y_min,y_max)


    #residuals (raw - gaussian)
    im3 = axes[2].pcolormesh(X2, Y2, residual_map, shading='auto') 
    axes[2].set_title("Residual Errors")
    axes[2].set_xlabel("Frequency Coordinates")
    fig.colorbar(im3, ax=axes[2])
    axes[2].set_xlim(x_min,x_max)
    axes[2].set_ylim(y_min,y_max)

    axes[0].set_aspect('equal', 'box')
    axes[1].set_aspect('equal', 'box')
    axes[2].set_aspect('equal', 'box')


    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    # plt.close()

def plot_integral_and_peak_normalization(x_min, x_max, y_min, y_max, difference_fft_integral,difference_fft_peak, save_path):
    #plotting side by side for integral and peak
    limit_integral = max(abs(difference_fft_integral.min()), abs(difference_fft_integral.max()))

    limit_peak = max(abs(difference_fft_peak.min()), abs(difference_fft_peak.max()))


    fig, ((diff1, diff2), (diff_phase1, diff_phase2)) = plt.subplots(2,2, figsize=(12,8))

    diff_integral = diff1.imshow(difference_fft_integral, extent=[x_min, x_max, y_min, y_max],cmap='bwr',vmin=-limit_integral,vmax=limit_integral)
    plt.colorbar(diff_integral, ax=diff1)
    diff1.set_title("Difference with integral Norm")
    diff1.set_xlim(-3,3)
    diff1.set_ylim(-3,3)

    diff_peak = diff2.imshow(difference_fft_peak, extent=[x_min, x_max, y_min, y_max],cmap='seismic', vmin=-limit_peak,vmax=limit_peak)
    plt.colorbar(diff_peak, ax=diff2)
    diff2.set_title("Difference with peak Norm")
    diff2.set_xlim(-3,3)
    diff2.set_ylim(-3,3)

    diff_phase_integral = diff_phase1.imshow(np.atan2(np.imag(difference_fft_integral), np.real(difference_fft_integral)), extent=[x_min, x_max, y_min, y_max],vmin=-limit_integral,vmax=limit_integral)
    diff_phase1.set_title("Difference phase for integral")
    diff_phase1.set_xlim(-3,3)
    diff_phase1.set_ylim(-3,3)

    diff_phase_peak = diff_phase2.imshow(np.atan2(np.imag(difference_fft_peak), np.real(difference_fft_peak)), extent=[x_min, x_max, y_min, y_max],vmin=-limit_peak,vmax=limit_peak)
    diff_phase2.set_title("Difference phase for peak")
    diff_phase2.set_xlim(-3,3)
    diff_phase2.set_ylim(-3,3)

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    # plt.close()

def plot_difference_between_ffts(x_min,x_max, y_min, y_max, difference_fft,I_far_original,I_far, save_path):
    #plotting the difference between far fields
    limit_integral = max(abs(difference_fft.min()), abs(difference_fft.max()))


    fig, ((orig, non, diff), (orig_phase, non_phase, diff_phase)) = plt.subplots(2,3, figsize=(12,8))

    orig1 = orig.imshow(np.log10(np.abs(I_far_original)**2), extent=[x_min, x_max, y_min, y_max])
    orig.set_title("Flat mirror FFT")
    orig.set_xlim(-3,3)
    orig.set_ylim(-3,3)

    non1 = non.imshow(np.log10(np.abs(I_far)**2), extent=[x_min, x_max, y_min, y_max])
    non.set_title("Non-flat mirror FFT")
    non.set_xlim(-3,3)
    non.set_ylim(-3,3)

    diff1 = diff.imshow(difference_fft_integral, extent=[x_min, x_max, y_min, y_max],vmin=-limit_integral,vmax=limit_integral)
    plt.colorbar(diff1, ax=diff)
    diff.set_title("Difference with integral Norm")
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

    diff_phase1 = diff_phase.imshow(np.atan2(np.imag(difference_fft_integral), np.real(difference_fft)), extent=[x_min, x_max, y_min, y_max],vmin=-limit_integral,vmax=limit_integral)
    diff_phase.set_title("Difference phase")
    diff_phase.set_xlim(-3,3)
    diff_phase.set_ylim(-3,3)

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    # plt.close()

def aperture_near_fft_plots(aperture, radius, near, length, width,I_far, x_min, x_max, y_min, y_max, save_path):
    #new printing layout six side by side
    fig, ((ap, close, far), (ap_phase, close_phase , far_phase)) = plt.subplots(2, 3, figsize=(8,12))

    ap1 = ap.imshow(np.abs(aperture), extent = [-radius, radius, -radius, radius])
    ap.set_title("Aperture Amplitude")
    ap.set_ylabel("Diameter of Aperture (m)")

    close1 = close.imshow(10 * (np.log10(np.abs(near)**2) - np.log10(np.abs(near).max()**2)), 
                extent=[length.min(), length.max(), width.min(), width.max()], 
                aspect='equal', 
                origin='lower')
    close.set_title("Mirror Projection Amplitude")
    close.set_xlabel("Length (m)")
    close.set_ylabel("Width (m)")

    far1 = far.imshow((np.log10(np.abs(I_far)**2)) , extent=[x_min, x_max, y_min, y_max])
    far.set_title("Far-field Amplitude")
    fig.colorbar(far1, ax=far, label="Intensity")
    far.set_xlabel("Degrees")
    far.set_ylabel("Degrees")
    # far.set_xlim(-3,3)
    # far.set_ylim(-3,3)

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
    # far_phase.set_xlim(-3,3)
    # far_phase.set_ylim(-3,3)

    plt.tight_layout()
    plt.savefig(save_path) ####savefig to save_path when save_path is == true or somethin
    # plt.show()
    # plt.close()

def mirror_plotting_cold():
    # mirror plotting
    Set_Cold = load_points(r"E:\BICEP\BICEP_MIRROR\BICEP_MIRROR\cold.txt")
    plot_points(Set_Cold, "Z (um)", -400, 1500, mirror_data_1[0],mirror_data_1[1],new_mirror_z[2])


aperture, x, y= ap_grid(N, dx, N_mirror, k, radius, edge_taper, phase_gradient)

mirror_data_1, length, width, distance = mirror(dx, theta_1, mirror_coord)

#for new phase output of non flat mirror
new_mirror_z = bilinear_interpolation(mirror_data_1)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#flat mirror
# near = fresnel(aperture, x, y, mirror_data_1, k)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! computational save
#most changed variables (N, theta_1, theta_2)
cache_filename = f"fresnel_near_field_N{N}N_mirror{N_mirror}_theta1{theta_1}_theta2{theta_2}.npy"

if os.path.exists(cache_filename):
    print("File already found. Loading in Fresnel near-field grid")
    near = np.load(cache_filename)
else:
    print("File not found. Running Fresnel calculation")
    near = fresnel(aperture, x, y, mirror_data_1, k)    
    np.save(cache_filename, near)
    print(f"Fresnel calculation saved to '{cache_filename}'")
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

non_flat = apply_phase_array_nonflat(k, near, new_mirror_z)

#non flat specs
I_far, normalize_integral, normalize_peak, x_degree, y_degree, X2, Y2 = fraunhofer(N_mirror,theta_1, k, non_flat, mirror_coord)

# check_x, check_y = X2, Y2 

#flat specs
I_far_original, normalize_orig_integral, normalize_orig_peak, x_degree, y_degree, X2, Y2 = fraunhofer(N_mirror,theta_1, k, near, mirror_coord)

#gaussian graphs change
change_type_of_analysis = I_far
type_of_mirror_analysis = "Non-flat"

#for non log plot 
# difference_fft = (normalize_orig - normalize).real
difference_fft_peak = (normalize_orig_peak - normalize_peak).real
difference_fft_integral = (normalize_orig_integral - normalize_integral).real



# preparation for 2d gauss
I_far_intensity = np.abs(change_type_of_analysis) ** 2


# inputs
xy_input = np.vstack((X2.ravel(), Y2.ravel()))
ydata = I_far_intensity.ravel()

# initial guesses based on data distribution
peak_height = float(np.max(I_far_intensity))
omega_guess = 1.0 / peak_height if peak_height != 0 else 1.0

# where the beam is strong to guess center shifts
total_mass = np.sum(I_far_intensity) if np.sum(I_far_intensity) > 0 else 1.0
mu_x_guess = np.sum(X2 * I_far_intensity) / total_mass
mu_y_guess = np.sum(Y2 * I_far_intensity) / total_mass

# Estimate physical beam widths 
sigma_x_guess = np.sqrt(np.sum((X2 - mu_x_guess)**2 * I_far_intensity) / total_mass)
sigma_y_guess = np.sqrt(np.sum((Y2 - mu_y_guess)**2 * I_far_intensity) / total_mass)

theta_guess = np.radians(theta_1) 
p0 = [omega_guess, mu_x_guess, mu_y_guess, sigma_x_guess, sigma_y_guess, theta_guess]

# physical bounds tracking
max_x_span = float(np.ptp(X2))
max_y_span = float(np.ptp(Y2))

lower_bounds = [0,          -max_x_span, -max_y_span, 1e-12,        1e-12,        -np.pi]
upper_bounds = [np.inf,      max_x_span,  max_y_span, max_x_span,   max_y_span,   np.pi]
bounds = (lower_bounds, upper_bounds)

popt, pcov = curve_fit(gaussian_fit, xdata=xy_input, ydata=ydata, p0=p0, bounds=bounds)
optimized_omega, opt_mu_x, opt_mu_y, opt_sigma_x, opt_sigma_y, opt_theta = popt

# physical variables to covariance matrix layout
cos_t = np.cos(opt_theta)
sin_t = np.sin(opt_theta)

# from principal widths to covariance metrics
optimal_xx = (opt_sigma_x**2 * cos_t**2) + (opt_sigma_y**2 * sin_t**2)
optimal_yy = (opt_sigma_x**2 * sin_t**2) + (opt_sigma_y**2 * cos_t**2)
# rotation layout
optimal_xy = (opt_sigma_x**2 - opt_sigma_y**2) * sin_t * cos_t 

popt_legacy = np.array([optimized_omega, opt_mu_x, opt_mu_y, optimal_xx, optimal_yy, optimal_xy])

#manually name for rewrite error
far_field = np.abs(change_type_of_analysis)**2

#exporting information about comparision graphs to a txt file
data_folder = r"C:\Users\lj350\Downloads\BICEP\BICEP_MIRROR\Data"

#add in flat vs non flat 33
data_run_name = f"fresnel_near_field_N{N}N_mirror{N_mirror}_theta1{theta_1}_theta2{theta_2}{type_of_mirror_analysis}{date_time}.txt"
data_name = os.path.join(data_folder, data_run_name)

# os.makedirs(data_name, exist_ok=True)

#optimized parameters then reshape
fit_map_flat = gaussian_fit(xy_input, *popt)
fit_map_2d = fit_map_flat.reshape(X2.shape)
residual_map = I_far_intensity - fit_map_2d

#axis adjustment 
# x_min, x_max = -x_degree, x_degree
x_min, x_max = float(X2.min()), float(X2.max())
y_min, y_max = float(Y2.min()), float(Y2.max())

#printing system creating the folder name for tasks
parent_folder = r"C:\Users\lj350\Downloads\BICEP\BICEP_MIRROR\Graphs"
run_name = f"fresnel_near_field_N{N}N_mirror{type_of_mirror_analysis}_theta1{theta_1}_theta2{theta_2}"
folder_name = os.path.join(parent_folder, run_name)

# os.makedirs(folder_name, exist_ok=True)

print(f"Graphs saved to: {folder_name}")

#create list of to call functions
tasks = [
(lambda path: plot_2d_gaussian(type_of_mirror_analysis, far_field, X2, Y2, fit_map_2d, residual_map, popt_legacy, path), f"Gaussian{date_time}"),

        
(lambda path: plot_integral_and_peak_normalization(x_min, x_max, y_min, y_max, difference_fft_integral,difference_fft_peak, path), f"Normalization{date_time}"),


(lambda path: plot_difference_between_ffts(x_min,x_max, y_min, y_max, difference_fft_integral,I_far_original,I_far, path),f"FFT_compare{date_time}"), 


(lambda path: aperture_near_fft_plots(aperture, radius, near, length, width,I_far, x_min, x_max, y_min, y_max, path),f"Plot Check{date_time}")

]

#call the functions to save them in the graphs folder
for function, plot_name in tasks:
    file_path = os.path.join(folder_name, f"{plot_name}.png")
    function(file_path)