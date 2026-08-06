import numpy as np
#insert before the for loop in fresnel
z = mirror_data[2]
X1, X2 = np.meshgrid(rows, cols)
Y1, Y2 = np.meshgrid(near_field[0].shape[0], near_field[1].shape[1])

#calculate physical distance between coordinate pairs 
distance_matrix = np.sqrt((X2- X1) ** 2 + (Y2 - Y1)**2 + z**2)

#oblique factor?
obq = z / distance_matrix

#clculation


######semi vectorized

# --- Memory-Safe Alternative: Eliminates the inner loop completely ---
x_flat = x.flatten()
y_flat = y.flatten()
aperture_flat = aperture.flatten()

rows, cols = mirror_data[0].shape[0], mirror_data[0].shape[1]
near_field = np.zeros((rows, cols), dtype=np.complex128)

# Pre-shape aperture arrays for 2D broadcasting: shape (1, N^2)
x_ap_2d = x_flat[np.newaxis, :]
y_ap_2d = y_flat[np.newaxis, :]
aperture_2d = aperture_flat[np.newaxis, :]

for j in tqdm(range(rows)):
    # Pull the entire j-th row of observation coordinates at once: shape (cols, 1)
    obs_x_col = mirror_data[0][j, :].reshape(-1, 1)
    obs_y_col = mirror_data[1][j, :].reshape(-1, 1)
    obs_dist_col = mirror_data[2][j, :].reshape(-1, 1)
    
    # Calculate distance for the entire row of observation points. Shape: (cols, N^2)
    d_2d = np.sqrt((x_ap_2d - obs_x_col)**2 + (y_ap_2d - obs_y_col)**2 + obs_dist_col**2)
    
    # Calculate kernel for the row
    kernel_2d = np.exp(1j * k * d_2d) / d_2d
    
    # Sum along axis 1 (the aperture axis) to fill the entire row of near_field instantly
    near_field[j, :] = np.sum(aperture_2d * kernel_2d, axis=1)




## numba interpret 
import numpy as np
from numba import njit, prange

# We isolate your core calculation loop and add the Numba decorator.
# 'parallel=True' automatically uses all your computer's CPU cores.
@njit(parallel=True, fastmath=True)
def calculate_near_field_numba(mirror_x, mirror_y, mirror_z, x_flat, y_flat, aperture_flat, k):
    rows, cols = mirror_x.shape
    num_ap = len(x_flat)
    near_field = np.zeros((rows, cols), dtype=np.complex128)
    
    # prange tells Numba to run these row calculations in parallel across CPU cores
    for j in prange(rows):     
        for i in range(cols): 
            obs_x = mirror_x[j, i]
            obs_y = mirror_y[j, i]
            obs_dist = mirror_z[j, i]
            
            # Simple, direct pixel loop with NO memory allocations
            total_sum = 0.0 + 0.0j
            for a in range(num_ap):
                d = np.sqrt((x_flat[a] - obs_x)**2 + (y_flat[a] - obs_y)**2 + obs_dist**2)
                total_sum += aperture_flat[a] * np.exp(1j * k * d) / d
                
            near_field[j, i] = total_sum
            
    return near_field

# =======================================================
# HOW TO RUN IT IN YOUR CODE:
# =======================================================
# Just flatten your aperture arrays once before calling it:
x_flat = x.flatten()
y_flat = y.flatten()
aperture_flat = aperture.flatten()

# Call the function (the first run takes 1-2 seconds to compile, then it is blazing fast!)
near_field = fresnel(mirror_data[0], mirror_data[1], mirror_data[2], 
    x_flat, y_flat, aperture_flat, k)




#orignial 
#rows, cols = t_x.shape[0], t_x.shape[1]
    # rows, cols = mirror_data[0].shape[0], mirror_data[0].shape[1]

    # # (Rows Cols)
    # near_field = np.zeros((rows, cols), dtype=np.complex128)

    # for j in tqdm(range(rows)):     
    #     for i in range(cols): 
    #         obs_x = mirror_data[0][j,i]
    #         obs_y = mirror_data[1][j,i]
    #         obs_dist = mirror_data[2][j,i]

    #         d = np.sqrt((x - obs_x)**2 + (y - obs_y)**2 + obs_dist**2)
    #         near_field[j, i] = (np.sum(aperture * np.exp(1j * k * d) / d))
