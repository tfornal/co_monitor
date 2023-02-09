__author__ = "T. Fornal"

from simulation import Simulation

"""Executes the simulation code in order to calculate 
plasma volume observed by each spectroscopic channel.
Requires to input list of elements (B, C, N or O)"""

elements_list = ["C"]#, "B" , "N", "O"]
testing_settings = dict(
    slits_number=10,
    distance_between_points=20,
    crystal_height_step=15,
    crystal_length_step=10,
    savetxt=False,
    plot=True,
)

if __name__ == "__main__":
    for element in elements_list:
            simul = Simulation(element, **testing_settings)