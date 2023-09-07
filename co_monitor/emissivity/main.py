__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from collections import namedtuple

from co_monitor.emissivity.kinetic_profiles import (
    TwoGaussSumProfile,
    ExperimentalProfile,
    TestExperimentalProfile,
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
# Select kinetic profiles
# kinetic_profiles = ExperimentalProfile("report_20181011_012@5_5000_v_1").profiles_df
# kinetic_profiles = TestExperimentalProfile("20230215.058").profiles_df


# 20230215.058
# 8.0105	8.5105	9.0105	9.5105	10.0105	10.5105	11.0105	11.5105	12.0105	12.5105	13.0105	13.5105	14.0105	14.5105	15.0105	15.5105	16.0105	16.511

# 20230216.028
# 2.1451000000000002	2.2451	2.3451	2.4451	2.5450999999999997	2.6451000000000002	2.7451	2.8451	2.9451	3.0450999999999997	3.1451000000000002	3.2451	3.3451	3.4451	3.5450999999999997	3.6451000000000002	3.7451	3.8451	3.9451	4.0451	4.1451	4.2451	4.3451	4.4451	4.5451	4.6451	4.7451	4.8451	4.9451	5.0451	5.1451	5.2451	5.3451	5.4451	5.5451	5.6451	5.7451	5.8451	5.9451	6.0451	6.1451	6.2451	6.3451	6.4451	6.5451	6.6451	6.7451	6.8451	6.9451	7.0451	7.1101

# 20230223.037
# 3.0906000000000002	3.2906	3.4905999999999997	3.6906	3.8906	4.0906	4.2905999999999995	4.490600000000001	4.6906	4.8906	5.0906	5.2905999999999995	5.490600000000001	5.6906	5.8906	6.0906	6.2905999999999995	6.490600000000001	6.6906	6.8906	7.0906	7.2905999999999995	7.490600000000001	7.6906	7.8906	8.0056

# shot = "20230215.058"
# time_frames = [
#     8.0105,
#     8.5105,
#     9.0105,
#     9.5105,
#     10.0105,
#     10.5105,
#     11.0105,
#     11.5105,
#     12.0105,
#     12.5105,
#     13.0105,
#     13.5105,
#     14.0105,
#     14.5105,
#     15.0105,
#     15.5105,
#     16.0105,
#     16.511,
# ]


# shot = "20230216.028"
# time_frames = [
#     2.1451000000000002,
#     2.2451,
#     2.3451,
#     2.4451,
#     2.5450999999999997,
#     2.6451000000000002,
#     2.7451,
#     2.8451,
#     2.9451,
#     3.0450999999999997,
#     3.1451000000000002,
#     3.2451,
#     3.3451,
#     3.4451,
#     3.5450999999999997,
#     3.6451000000000002,
#     3.7451,
#     3.8451,
#     3.9451,
#     4.0451,
#     4.1451,
#     4.2451,
#     4.3451,
#     4.4451,
#     4.5451,
#     4.6451,
#     4.7451,
#     4.8451,
#     4.9451,
#     5.0451,
#     5.1451,
#     5.2451,
#     5.3451,
#     5.4451,
#     5.5451,
#     5.6451,
#     5.7451,
#     5.8451,
#     5.9451,
#     6.0451,
#     6.1451,
#     6.2451,
#     6.3451,
#     6.4451,
#     6.5451,
#     6.6451,
#     6.7451,
#     6.8451,
#     6.9451,
#     7.0451,
#     7.1101,
# ]

shot = "20230223.037"
time_frames = [
    3.0906000000000002,
    3.2906,
    3.4905999999999997,
    3.6906,
    3.8906,
    4.0906,
    4.2905999999999995,
    4.490600000000001,
    4.6906,
    4.8906,
    5.0906,
    5.2905999999999995,
    5.490600000000001,
    5.6906,
    5.8906,
    6.0906,
    6.2905999999999995,
    6.490600000000001,
    6.6906,
    6.8906,
    7.0906,
    7.2905999999999995,
    7.490600000000001,
    7.6906,
    7.8906,
    8.0056,
]


# for time in time_frames:
#     kinetic_profiles = TestExperimentalProfile(f"{shot}", str(time)).profiles_df
#     for element in lyman_alpha_lines:
#         line = lyman_alpha_line[element]
#         ce = Emissivity(
#             element,
#             line.ion_state,
#             line.wavelength,
#             line.impurity_concentration,
#             transitions,
#             reff_magnetic_config,
#             kinetic_profiles,
#             time,
#         )
#         ce.plot(savefig=False)


# ==========================================
n_e = [4e13, 0, 0.937, 1.8e12, 0.5, 0.11]
T_e1 = [500, 0, 0.175, 30, 0.38, 0.07]
T_e2 = [10000, 0, 0.175, 1500, 0.38, 0.07]
import numpy as np

T_e = np.linspace(T_e1, T_e2, 2)
print(T_e)
for idx, i in enumerate(T_e):
    kinetic_profiles = TwoGaussSumProfile(n_e, i, plot=True).profiles_df
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
