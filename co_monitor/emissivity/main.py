__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from collections import namedtuple

import numpy as np

from co_monitor.emissivity.kinetic_profiles import (
    TwoGaussSumProfile,
    ExperimentalProfile,
    TestExperimentalProfile,
)
from co_monitor.emissivity.pec import PEC
from co_monitor.emissivity.reader import Emissivity

lyman_alpha_lines = [
    "O",
    # "C",
    # "N",
    # "B",
]
Element = namedtuple("Element", "ion_state wavelength impurity_concentration")

lyman_alpha_line = {
    "B": Element("Z4", 48.6, 2),
    "C": Element("Z5", 33.7, 2),
    "N": Element("Z6", 24.8, 2),
    "O": Element("Z7", 19.0, 2),
    # "O": Element("Z6", 18.6, 2),
}
transitions = ["EXCIT", "RECOM"]
reff_magnetic_config = "Reff_coordinates-10_mm"
# Select kinetic profiles
# kinetic_profiles = ExperimentalProfile("report_20181011_012@5_5000_v_1").profiles_df
# kinetic_profiles = TestExperimentalProfile("20230215.058").profiles_df


# ==========================================
n_e = [4e13, 0, 0.937, 1.8e12, 0.5, 0.11]
T_e1 = [500, 0, 0.175, 30, 0.38, 0.07]
T_e2 = [10000, 0, 0.175, 1500, 0.38, 0.07]

T_e = np.linspace(T_e1, T_e2, 11)
for idx, i in enumerate(T_e):
    kinetic_profiles = TwoGaussSumProfile(n_e, i, plot=False).profiles_df
    time = idx
    for element in lyman_alpha_lines:
        line = lyman_alpha_line[element]
        ce = Emissivity(
            element,
            line.ion_state,
            line.wavelength,
            line.impurity_concentration,
            transitions,
            reff_magnetic_config,
            kinetic_profiles,
            time,
        )
        ce.plot(savefig=False)
