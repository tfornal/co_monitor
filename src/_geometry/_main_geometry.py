import time

from simulation import Simulation

"""Executes the simulation code in order to calculate 
plasma volume observed by each spectroscopic channel.
Requires to input list of elements (B, C, N or O)"""

elements_list = ["B", "C"]  # , "C", "O"]
testing_settings = dict(
    slits_number=10,
    distance_between_points=50,
    crystal_height_step=5,
    crystal_length_step=5,
    savetxt=False,
    plot=True,
)

if __name__ == "__main__":
    for element in elements_list:
        simul = Simulation(element, **testing_settings)
