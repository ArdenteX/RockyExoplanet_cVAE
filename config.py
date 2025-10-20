import numpy as np
np.random.seed(125)


class PlanetsDataGenerator:
    def __init__(self, size):
        self.size = size

    def Kepler78b(self):
        mass_samples_noise = np.random.normal(1.77, 0.25, size=self.size)
        radius_samples_noise = np.random.normal(1.228, 0.019, size=self.size)
        femg_samples_noise = np.random.normal(0.813, 0.248, size=self.size)
        simg_samples_noise = np.random.normal(0.933, 0.281, size=self.size)
        kepler78b = np.stack([mass_samples_noise, radius_samples_noise, femg_samples_noise, simg_samples_noise], axis=1)
        return kepler78b

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

