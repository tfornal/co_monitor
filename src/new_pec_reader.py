import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import islice
import numpy as np
import scipy


class PEC2:
    def __init__(
        self,
        element,
        wavelength,
        transition,
        interp_step=20,
        plot=True,
    ):

        self.element = element
        self.wavelength = wavelength
        self.transition = transition
        self.interp_step = interp_step
        self.file_path = self.get_file_path()
        self.head_idx, self.ne_nodes_nr, self.te_nodes_nr = self._get_basic_info()
        self.pec_data_array, self.ne_nodes, self.te_nodes = self._read_data()
        self.interpolated_pec = self.interpolation()
        if plot:
            self.plot_pec_data()

    def get_file_path(self):
        pec_element_path = (
            Path(__file__).parent.resolve() / "_Input_files" / "PEC" / self.element
        )
        file_path = list(Path(pec_element_path).glob("*.dat"))[0]
        return file_path

    def _get_basic_info(self):
        with open(self.file_path) as file:
            self.data = file.readlines()
        for idx, line in enumerate(self.data):
            try:
                splitted_headline = line.split()
                if (float(splitted_headline[0]) == float(self.wavelength)) & (
                    self.transition in splitted_headline
                ):
                    splitted_headline = line.split()
                    head = splitted_headline
                    head_idx = idx + 1
            except ValueError:  # End of the data. Comments section later on.
                break
        ne_nodes_nr, te_nodes_nr = head[2], head[3]

        return head_idx, int(ne_nodes_nr), int(te_nodes_nr)

    def _read_data(self):
        nr_of_rows = int(
            np.ceil(self.ne_nodes_nr / 8)
            + np.ceil(self.te_nodes_nr / 8)
            + self.ne_nodes_nr * (np.ceil(self.te_nodes_nr / 8))
        )
        readed_data = []
        with open(self.file_path) as fp:
            for line in islice(fp, self.head_idx, self.head_idx + nr_of_rows):
                readed_data.extend(line.split())

        data_array = np.array(readed_data).astype(np.float64)
        ne_nodes = data_array[: self.ne_nodes_nr]
        te_nodes = data_array[self.ne_nodes_nr : self.ne_nodes_nr + self.te_nodes_nr]
        pec_data_array = data_array[self.ne_nodes_nr + self.te_nodes_nr :].reshape(
            (self.ne_nodes_nr, self.te_nodes_nr)
        )

        return pec_data_array, ne_nodes, te_nodes

    def interpolation(self):
        fvals = self.pec_data_array.T
        func = interpolate.interp2d(self.ne_nodes, self.te_nodes, fvals, kind="linear")
        ne_new = np.linspace(
            self.ne_nodes[0], self.ne_nodes[-1], self.interp_step, endpoint=True
        )
        te_new = np.linspace(
            self.te_nodes[0], self.te_nodes[-1], self.interp_step, endpoint=True
        )
        pec_new_arr = func(ne_new, te_new)
        ne_new_arr, te_new_arr = np.meshgrid(ne_new, te_new)
        interpolated_pec = np.stack((ne_new_arr, te_new_arr, pec_new_arr), axis=0)

        return interpolated_pec

    def plot_pec_data(self):
        ne, te, pec = self.interpolated_pec
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(ne, te, pec, rstride=1)
        plt.title(f"PEC - {transition}")
        ax.set_xlabel("Ne [1/cm3]")
        ax.set_ylabel("Te [eV]")
        ax.set_zlabel("PEC")
        plt.show()


if __name__ == "__main__":
    transitions_list = ["EXCIT", "RECOM"]
    lista = []
    for transition in transitions_list:
        pec = PEC("C", 33.7, transition, interp_step=10, plot=False)
        trans = pec.interpolated_pec
        lista.append(trans)
        # pec = PEC("B", 194.3, transition, interp_step=10, plot=False)
        # pec = PEC("O", 102.4, transition, interp_step=10, plot=False)
        # pec = PEC("N", 133.8, transition, interp_step=10, plot=False)

    x = np.array(lista)
    print(x)
