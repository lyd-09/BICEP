import matplotlib.pyplot as plt
import numpy as np

class Field:
    def __init__(self, N, dx, wavelength ):
        self.N = N
        self.dx = dx
        self.wavelength = wavelength

        #complex field
        self.u = np.zeros(shape=(N,N),dtype=complex)

        #coordinate grid (assuming equal spacing in x and y)
        x = (np.arange(N) - N/2) * dx
        self.X, self.Y = np.meshgrid(x,x)

def aperture(field, radius, edge_taper):
    r = np.sqrt(field.X**2 + field.Y**2)
    aperture = np.zeros_like(field.u)
    #edge taper
    alpha = -np.log(edge_taper)/radius**2

    #Creates circular aperture
    inside = r < radius
    aperture[inside] = np.exp(-alpha * r[inside] ** 2)

    field.u = aperture.astype(complex)

    return field

def fresnel(field, z):
    k = 2*np.pi / field.wavelength
    N = field.N
    dx = field.dx

    near_field = Field(N,dx, field.wavelength)

    for i in range(N):
        for j in range(N):

            d = np.sqrt((field.X - near_field.X[i,j])**2 + (field.Y - near_field.Y[i,j])**2 + z**2)

            near_field.u[i,j] = np.sum(field.u * np.exp(1j * k * d) / d)
    return near_field

def fraunhofer(field):

    center = np.fft.ifftshift(field.u)

    centered = np.fft.fft2(center)
    centered = np.fft.fftshift(centered)

    #scale for dx (area element)
    centered *= field.dx**2

    #angular axes (same centered)
    fx = np.fft.fftshift(np.fft.fftfreq(field.N, d=field.dx)) 
    fy = np.fft.fftshift(np.fft.fftfreq(field.N, d=field.dx))

    Xf, Yf = np.meshgrid(fx, fy)

    far_field = Field(field.N, field.dx, field.wavelength)
    far_field.u = centered
    far_field.X = Xf
    far_field.Y = Yf

    return far_field

def embed(field, embedded_size):
    embedded = Field(embedded_size, field.dx, field.wavelength)
    #centering
    start = (embedded_size - field.N) // 2
    end = start + field.N
    #placing
    embedded.u[start:end, start:end] = field.u

    return embedded


###
N = 100
L = 0.5 #500 mm across
dx = L/N
wavelength = 3e-3 #3mm
###

field = Field(N, dx, wavelength)
# plt.imshow(field.X)
# plt.colorbar()
# plt.show()

#aperture with 250 mm radius
field = aperture(field, radius=0.25, edge_taper=.01)
# plt.imshow(np.abs(field.u))
# plt.colorbar()
# plt.show()


#fresnel
# near = fresnel(field, z= 15)
# plt.imshow(np.log(np.abs(field.u)))
# plt.colorbar()
# plt.show()

#fraunhofer
far = fraunhofer(field)
plt.imshow(np.abs(far.u))
plt.colorbar()
plt.show()

#embedded
# bigger_grid = embed(field, 512)
# far_higher_res = fraunhofer(bigger_grid)
# plt.imshow(np.abs(far_higher_res.u))
# plt.colorbar()
# plt.show()



#new? 

#aperture.u,x,y,z

#mirror.u,x,y,z

#freshnel(aperture, mirror, lamda)

