__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from collections import namedtuple

# from pathlib import Path, PurePath
# import time
# from typing import List

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pyvista as pv

# from co_monitor.emissivity.fractional_abundance import FractionalAbundance
from co_monitor.emissivity.kinetic_profiles import (
    TwoGaussSumProfile,
    ExperimentalProfile,
)

from co_monitor.emissivity.pec import PEC
from co_monitor.emissivity.reader import Emissivity

lyman_alpha_lines = ["O"]  # , "C", "N", "O"]
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
n_e = [4e13, 0, 0.37, 9.8e12, 0.5, 0.11]
T_e = [570, 0, 0.155, 210, 0.38, 0.07]

# Select kinetic profiles
# kinetic_profiles = ExperimentalProfile("report_20181011_012@5_5000_v_1").profiles_df
kinetic_profiles = TwoGaussSumProfile(n_e, T_e, plot=True).profiles_df

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
    )
    ce.plot(savefig=True)
