import pytest

# from snapshottest.pytest import snapshottest
from collections import namedtuple
from co_monitor.emissivity.reader import Emissivity
from co_monitor.emissivity.kinetic_profiles import TwoGaussSumProfile

# lyman_alpha_lines = ["B"]  # , "C", "N", "O"]
Element = namedtuple("Element", "ion_state wavelength impurity_concentration")

lyman_alpha_line = {
    "B": Element("Z4", 48.6, 2),
    "C": Element("Z5", 33.7, 2),
    "N": Element("Z6", 24.8, 2),
    "O": Element("Z7", 19.0, 2),
}
transitions = ["EXCIT", "RECOM"]
reff_magnetic_config = "Reff_coordinates-10_mm"
n_e = [7e13, 0, 0.37, 9.8e12, 0.5, 0.11]
T_e = [1870, 0, 0.155, 210, 0.38, 0.07]

# Select kinetic profiles
# kinetic_profiles = ExperimentalProfile("report_20181011_012@5_5000_v_1").profiles_df
kinetic_profiles = TwoGaussSumProfile(n_e, T_e).profiles_df

data_container = {}
line = lyman_alpha_line["C"]


def test_emissivity():
    emissivity = Emissivity(
        "C",
        line.ion_state,
        line.wavelength,
        line.impurity_concentration,
        transitions,
        reff_magnetic_config,
        kinetic_profiles,
    )
    snapshot.assert_match(emissivity)
    # assert emissivity ==
