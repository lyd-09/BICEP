def fraunhofer(lamda, window_size, z1, near_grid, mirror_coord):

    #scaling difference for far field (should be the physical width of the source grid/ number of rows in ) !!!!!! global variable? (possibly use 'dy' as in frequency (fft) spatial coords to real space)

    rows_in_mirror = near_grid.shape[0]
    col_mirror = near_grid.shape[1]
    wl = 2.0
    k = 2 * np.pi / 2

    dx = mirror_coord[0] / col_mirror
    dy = mirror_coord[1] / rows_in_mirror

    #!!!!!!!!!!!!!
    mirror_dim = [2700,1800]
    mirror_res = [10,10]
    # Tilt by 45 degrees
    mirror_tilt = 45.0
    # Center at (x,y,z) = (0,0,4m)
    mirror_cen = [0.0, 0.0, 4000.0]
    (x1, y1) = np.meshgrid(np.arange(-mirror_dim[0] / 2.0, +mirror_dim[0] / 2.0, mirror_res[0]), 
                        np.arange(-mirror_dim[1] / 2.0, +mirror_dim[1] / 2.0, mirror_res[1]))
    z1 = y1 * np.sin(np.radians(mirror_tilt)) + mirror_cen[2]
    y1 = y1 * np.cos(np.radians(mirror_tilt)) + mirror_cen[1]
    x1 = x1 + mirror_cen[0]

    f1 = np.zeros(x1.shape, dtype=complex)

    npad = [2048, 2048]
    (x2, y2) = np.meshgrid(np.linspace(-npad[0]/2*mirror_res[0], +npad[0]/2*mirror_res[0], npad[0]), 
                        np.linspace(-npad[1]/2*mirror_res[1], +npad[1]/2*mirror_res[1], npad[1]))
    (x4, y4) = np.meshgrid(np.linspace(-2048*1, 2048*1, 4096), np.linspace(-2048*1, 2048*1, 4096))
    z4 = np.zeros(x4.shape)
    z2 = y2 * np.sin(np.pi / 4) + mirror_cen[2]
    y2 = y2 * np.cos(np.pi / 4) + mirror_cen[1]
    x2 = x2 + mirror_cen[0]
    f2 = np.zeros(shape=(npad[0],npad[1]), dtype=complex) #field complex amplitudes in mirror
    ix = (f2.shape[0] - f1.shape[0]) // 2
    iy = (f2.shape[1] - f1.shape[1]) // 2
    f2[ix:ix+f1.shape[0],iy:iy+f1.shape[1]] = f1

    f3 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f2 * np.exp(-1j * k * y2))))
    
    uv_width = [np.degrees(0.5 * wl / (x2[0,1] - x2[0,0])), np.degrees(0.5 * wl / (y2[1,0] - y2[0,0]))]
    f4 = np.zeros(shape=(4096,4096), dtype=complex)
    f5 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f4 * np.exp(1j * wl * z4))))
    uv_width = np.degrees(0.5 * wl / 1)



    #!!!!!!!!!!!

    # # #create arrays spanning from negative half-width to positive half-width
    # x_pixels = (np.arange(col_mirror) - (col_mirror - 1) / 2) * dx
    # y_pixels = (np.arange(rows_in_mirror) - (rows_in_mirror - 1) / 2) * dy

    # x2, y2 = np.meshgrid(x_pixels, y_pixels)

    # # #!!!! clean up the padding (double padding? 360 error)
    
    # # #pad the y2 for the phase shift propogation
    # # ix = (2048 - col_mirror) // 2  # Row index offset
    # # iy = (2048 - rows_in_mirror) // 2  # Column index offset


    # # xy2pad = np.zeros(shape=(2048, 2048), dtype=complex)

    # # xy2pad[ix : ix + col_mirror, iy : iy + rows_in_mirror]

    # # y2_component = xy2pad[:, iy : iy + rows_in_mirror]

    # # x2_component = xy2pad[:,ix:ix + col_mirror]

    # # #rotate around the x-axis by 45 degrees and shift to center
    # # z2 = xy2pad * np.sin(np.pi / 4) + 4000
    # # y2 = y2_component * np.cos(np.pi / 4) + 0
    # # x2 = x2_component + 0

    # # #!!!!!!!!!!
    
    # # #padding
    # # npad = [2048, 2048] 
    # # f2 = np.zeros(shape=(npad[0], npad[1]), dtype=complex)
    # # #to center
    # # ix = (f2.shape[0] - near_grid.shape[0]) // 2  # Row offset
    # # iy = (f2.shape[1] - near_grid.shape[1]) // 2  # Column offset

    # # # row and col dim
    # # f2[ix:ix + near_grid.shape[0], iy:iy + near_grid.shape[1]] = near_grid

    # # near_grid = f2

    # # 2. Calculate row and column padding for a 2048x2048 target
    # # y2 and x2 have shapes (rows_in_mirror, col_mirror) due to meshgrid
    # pad_rows = 2048 - rows_in_mirror
    # pad_cols = 2048 - col_mirror

    # # 3. Distribute padding evenly to the top/bottom and left/right
    # # (before, after) padding pairs
    # pad_y = (pad_rows // 2, pad_rows - (pad_rows // 2))
    # pad_x = (pad_cols // 2, pad_cols - (pad_cols // 2))

    # # 4. Apply the padding (filling the outer area with 0)
    # x2_padded = np.pad(x2, (pad_y, pad_x), mode='constant', constant_values=0)
    # y2_padded = np.pad(y2, (pad_y, pad_x), mode='constant', constant_values=0)

    # # Resulting shapes will be exactly (2048, 2048)

    # #wave number
    # k = 2 * np.pi / lamda

    # # X_spatial, Y_spatial = np.meshgrid(mirror_coord[0], mirror_coord[1])

    # z2 = np.ones_like(x2_padded) * 4000

    # angle = np.pi / 4  # 45 degrees

    # z2 = y2_padded * np.sin(angle) + 4000
    # phase = k * (z2 - 4000)
    # amplitude = np.zeros((2048, 2048))
    # pad_rows = (2048 - rows_in_mirror) // 2
    # pad_cols = (2048 - col_mirror) // 2
    # amplitude[pad_rows : pad_rows + rows_in_mirror, pad_cols : pad_cols + col_mirror] = 1.0
    # #phase ramp tilt
    # #for mirror non flatness
    # # phase_shift = np.exp((1j * 4*np.pi * (z2/lamda))/np.sin(theta_1))
    # # fraunhoffer_phase_factor = np.exp(1j * k * y2_padded)
    # angle_rad = np.radians(45)
    # fraunhoffer_phase_factor = np.exp(1j * k * y2_padded * np.sin(angle_rad))

    # wavefront = fraunhoffer_phase_factor * amplitude
    # # #setting up variables for FFT far field (obs = observational)
    # # #spatial freq for x and y 
    # # fx = np.fft.fftshift(np.fft.fftfreq(near_grid.shape[1], d=dx))
    # # fy = np.fft.fftshift(np.fft.fftfreq(near_grid.shape[0], d=dy))
    # # X2, Y2 = np.meshgrid(fx, fy)

    # # #convert to cos
    # # alpha = X2 * lamda
    # # beta = Y2 * lamda

    # # x_deg = np.arcsin(alpha) * (180/ np.pi)
    # # y_deg = np.arcsin(beta) * (180/np.pi)
    # # 1. Calculate the spatial frequency step sizes
    # df_x = 1 / (2048 * dx)
    # df_y = 1 / (2048 * dy)

    # # 2. Generate the 1D frequency coordinate arrays
    # fx = (np.arange(2048) - 2048 / 2) * df_x
    # fy = (np.arange(2048) - 2048 / 2) * df_y

    # # 3. Convert frequencies to directional cosines (alpha, beta)
    # # Note: Use your physical wavelength 'wavelength' here
    # alpha = fx * lamda
    # beta = fy * lamda

    # # 4. Check for invalid boundaries before arcsin to prevent NaN errors
    # # If your spatial sampling (dx, dy) is very small, alpha/beta could theoretically exceed 1.0
    # alpha = np.clip(alpha, -1.0, 1.0)
    # beta = np.clip(beta, -1.0, 1.0)

    # # 5. Convert directional cosines into degrees from the center
    # x_deg = np.arcsin(alpha) * (180 / np.pi)
    # y_deg = np.arcsin(beta) * (180 / np.pi)

    # # #complex amplitude scaling factor (quadratic phase factor)
    # # c = 1/(1j*lamda * z1) * np.exp(1j*k/(2*z1) * (X2 ** 2 + Y2 ** 2))

    # # #wave transformation 
    # obs_plane = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(wavefront)))

    # # obs_plane_field = obs_plane * dx * dy

    # #!!!!!!
    # # Calculate your angular width safely without dividing by zero
    # uv_width = [
    #     np.degrees(0.5 * lamda / dx), 
    #     np.degrees(0.5 * lamda / dy)
    # ]


    return f5, uv_width