import matplotlib.pyplot as plt
import numpy as np


#creating the aperture space
N = 100
aperture = np.zeros(shape=(N,N),dtype=complex)

x_pos = np.linspace(-250, 250, N) # 500 millimeter aperture
y_pos = np.linspace(-250, 250, N)

x_grid,y_grid = np.meshgrid(x_pos,y_pos)

#Computes distance from origin to each grid point
r = np.sqrt(x_grid**2 + y_grid**2)

#Creates circular aperture
aperture[r < 250] = 1.0

# """for i in range(len(x_pos)):
#     for j in range(len(y_pos)):
#         r = np.sqrt(x_pos[i]**2 + y_pos[j]**2)
#         if r < 250:
#             aperture[i,j] = 1.0"""


# plt.imshow(np.abs(aperture))


#taking the aperture and seeing where thats projected onto the near field
near_field = np.zeros(shape=(100,100), dtype=complex)
x2 = np.linspace(-1000,1000,100)
y2 = np.linspace(-1000,1000,100)
z = 10000

x_grid2,y_grid2 = np.meshgrid(x2,y2)

for i in range(len(x2)):
    for j in range(len(y2)):
        #finding the distance between the whole aperture and points in a plane
        d = np.sqrt((x_grid - x2[i])**2 + (y_grid - y2[j])**2 + z**2) #2D array of distances 
        near_field[i,j] = np.sum(aperture * np.exp(complex(0,1) * d/3)) #(3) scaling factor for the phase

plt.imshow(np.abs(near_field))

# # --- ADD THESE LINES TO THE BOTTOM ---

# --- REPLACE YOUR LAST 3 LINES WITH THESE ---

# 1. Normalize the data so the peak is 1.0 (Fixes the "thousands" problem)
final_map = np.abs(near_field)
final_map = final_map / np.max(final_map)

# 2. Plot without axes and with a clean colorbar
plt.imshow(final_map)
plt.title('Simulated Fresnel Beam Map')
plt.axis('off') # Keeps it clean like the BICEP maps
plt.colorbar()

plt.show()