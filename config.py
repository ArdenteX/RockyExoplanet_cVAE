import numpy as np
np.random.seed(125)


class PlanetsDataGenerator:
    def __init__(self, size):
        self.size = size

    def Kepler10b(self):
        mass_samples_10b = np.random.normal(3.72, 0.42, size=self.size)
        radius_samples_10b = np.random.normal(1.470, 0.03, size=self.size)
        femg_samples_10b = np.random.normal(0.589, 0.073, size=self.size)
        simg_samples_10b = np.random.normal(0.661, 0.098, size=self.size)

        kepler10b = np.stack([mass_samples_10b, radius_samples_10b, femg_samples_10b, simg_samples_10b], axis=1)
        return kepler10b


    def Kepler78b(self):
        mass_samples_noise = np.random.normal(1.77, 0.25, size=self.size)
        radius_samples_noise = np.random.normal(1.228, 0.019, size=self.size)
        femg_samples_noise = np.random.normal(0.813, 0.202, size=self.size)
        simg_samples_noise = np.random.normal(0.933, 0.262, size=self.size)
        kepler78b = np.stack([mass_samples_noise, radius_samples_noise, femg_samples_noise, simg_samples_noise], axis=1)
        return kepler78b


    def HD3167b(self):
        mass_samples_noise = np.random.normal(5.69, 0.44, size=self.size)
        radius_samples_noise = np.random.normal(1.574, 0.054, size=self.size)
        femg_samples_noise = np.random.normal(0.692, 0.103, size=self.size)
        simg_samples_noise = np.random.normal(0.692, 0.130, size=self.size)
        hd3167b = np.stack([mass_samples_noise, radius_samples_noise, femg_samples_noise, simg_samples_noise], axis=1)
        return hd3167b


    def HD219134b(self):
        mass_samples_noise = np.random.normal(4.74, 0.19, size=self.size)
        radius_samples_noise = np.random.normal(1.602, 0.055, size=self.size)
        femg_samples_noise = np.random.normal(0.794, 0.154, size=self.size)
        simg_samples_noise = np.random.normal(0.871, 0.195, size=self.size)

        hd219134b = np.stack([mass_samples_noise, radius_samples_noise, femg_samples_noise, simg_samples_noise], axis=1)
        return hd219134b


    def EPIC249893012b(self):
        mass_samples_noise = np.random.normal(8.75, 1.09, size=self.size)
        radius_samples_noise = np.random.normal(1.95, 0.09, size=self.size)
        femg_samples_noise = np.random.normal(0.692, 0.124, size=self.size)
        simg_samples_noise = np.random.normal(0.741, 0.134, size=self.size)

        epic249893012b = np.stack([mass_samples_noise, radius_samples_noise, femg_samples_noise, simg_samples_noise], axis=1)
        return epic249893012b

    def Earth(self):
        mass_samples_noise = np.random.normal(1.00, 0.02, size=self.size)
        radius_samples_noise = np.random.normal(1.00, 0.02, size=self.size)
        femg_samples_noise = np.random.normal(0.900, 0.018, size=self.size)
        simg_samples_noise = np.random.normal(0.800, 0.016, size=self.size)

        earth = np.stack([mass_samples_noise, radius_samples_noise, femg_samples_noise, simg_samples_noise], axis=1)
        return earth

input_parameters = [
    'Mass',
    'Radius',
    'FeMg',
    'SiMg',
]

output_parameters = [
    'WRF',
    'MRF',
    'CRF',
    'WMF',
    'CMF',
    'CPS',
    'CTP',
    'k2'
]

x_labels = [
    'WRF',
    'MRF',
    'CRF',
    'WMF',
    'CMF',
    '$P_{\mathrm{CMB}}$(TPa)',
    '$T_{\mathrm{CMB}}(10^{3}$K$)$',
    '$k_{2}$'
]
xlocators = [0.05, 0.2, 0.2, 0.02, 0.2, 0.4, 1, 0.2]

xminorlocators = [0.01, 0.04, 0.04, 0.004, 0.04, 0.08, 0.2, 0.04]

DEVICE = 'cuda'

BASE_PATH = '.\\data'

