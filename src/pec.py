__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from collections import namedtuple
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


class PEC:
    """A class for reading, storing and interpolating Photon Emissivity Coefficients (PEC) data."""

    def __init__(self, element, wavelength, transition, interp_step=2000, plot=False):
        """
        Parameters
        ----------
        element : str
            The name of the considered element for which PEC data will be read. Calculates only B, C, and O Lyman-alpha transitions.
        wavelength : float
            The wavelength in Angstroms of the considered Lyman-alpha line.
        transition : str
            The type of transition - excitation ("EXCIT") or recombination ("RECOM").
        interp_step : int, optional
            The number of interpolation points in each dimension.
            The default is 2000 (assuring reasonable precision of further simulations).
        plot : bool, optional
            Plots the PEC values for the given ne and te matrix in 3D, by default False.

        """
        self.element = element
        self.wavelength = wavelength
        self.transition = transition
        self.interp_step = interp_step
        self.file_path = self._get_file_path()
        self.head_idx, self.ne_nodes_nr, self.te_nodes_nr = self._get_header_info()
        self.pec_data_array, self.ne_nodes, self.te_nodes = self._read_data()
        self.interpolated_pec = self._interpolate_pec()
        if plot:
            self.plot_pec_data()

    def _get_file_path(self) -> Path:
        """
        Get the file path of the PEC data.

        Returns
        -------
        file_path : pathlib.Path
            Path to the PEC data file.
        """
        pec_path = (
            Path(__file__).parent.resolve() / "_Input_files" / "PEC" / self.element
        )
        file_path = next(Path(pec_path).glob("*.dat"))
        return file_path

    def _get_header_info(self):
        """
        Extract header information from the data file.

        Returns
        -------
        head_idx : int
            The starting index of the data in the file.
        ne_nodes_nr : int
            The number of electron density nodes.
        te_nodes_nr : int
            The number of electron temperature nodes.
        """
        with open(self.file_path) as file:
            self.data = file.readlines()
        for idx, line in enumerate(self.data):
            try:
                line_items = line.split()
                if (float(line_items[0]) == self.wavelength) and (
                    self.transition in line_items
                ):
                    head = line_items
                    head_idx = idx + 1
                    break
            except ValueError:  # End of the data. Comments section later on.
                break
        ne_nodes_nr, te_nodes_nr = map(int, (head[2], head[3]))
        return head_idx, ne_nodes_nr, te_nodes_nr

    def _read_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read and process the data from the file.

        Returns
        -------
        pec_data_array : np.ndarray
            A 2D array containing the processed PEC data.
        ne_nodes : np.ndarray
            A 1D array containing the electron density (ne) nodes.
        te_nodes : np.ndarray
            A 1D array containing the electron temperature (te) nodes.
        """
        nr_of_rows = int(
            np.ceil(self.ne_nodes_nr / 8)
            + np.ceil(self.te_nodes_nr / 8)
            + self.ne_nodes_nr * (np.ceil(self.te_nodes_nr / 8))
        )
        with open(self.file_path, "r") as file:
            readed_data = [
                val
                for line in islice(file, self.head_idx, self.head_idx + nr_of_rows)
                for val in line.split()
            ]
        data_array = np.array(readed_data, dtype=np.float64)
        ne_nodes = data_array[: self.ne_nodes_nr]
        te_nodes = data_array[self.ne_nodes_nr : self.ne_nodes_nr + self.te_nodes_nr]
        pec_data_array = data_array[self.ne_nodes_nr + self.te_nodes_nr :].reshape(
            (self.ne_nodes_nr, self.te_nodes_nr)
        )
        return pec_data_array, ne_nodes, te_nodes

    def _interpolate_pec(self) -> np.ndarray:
        """
        Interpolate the PEC data to obtain a higher resolution grid.

        Returns
        -------
        interpolated_pec : np.ndarray
            Interpolated PEC data with shape (N, M, 3), where N and M are the number of new
            electron density (ne_new) and electron temperature (te_new) nodes, respectively.
        """
        fvals = self.pec_data_array.T

        # Use 2D linear interpolation to obtain a higher resolution grid
        func = interpolate.interp2d(self.ne_nodes, self.te_nodes, fvals, kind="linear")

        # Create new nodes for electron density and electron temperature
        ne_new = np.linspace(
            self.ne_nodes[0], self.ne_nodes[-1], self.interp_step, endpoint=True
        )
        te_new = np.linspace(
            self.te_nodes[0], self.te_nodes[-1], self.interp_step, endpoint=True
        )

        # Interpolate the PEC data
        pec_new_arr = func(ne_new, te_new)

        # Create arrays for the new electron density and electron temperature
        ne_new_arr, te_new_arr = np.meshgrid(ne_new, te_new)

        # Combine the new electron density, electron temperature, and PEC data into a single array
        interpolated_pec = np.stack(
            (ne_new_arr.T.ravel(), te_new_arr.T.ravel(), pec_new_arr.T.ravel()), axis=0
        ).T
        interpolated_pec = interpolated_pec.reshape(-1, len(te_new), 3)

        return interpolated_pec

    def plot_pec_data(self):
        X = self.interpolated_pec[:, :, 0]
        Y = self.interpolated_pec[:, :, 1]
        Z = self.interpolated_pec[:, :, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(X, Y, Z, rstride=1)
        plt.title(f"PEC - {transition}")
        ax.set_xlabel("Ne [1/cm3]")
        ax.set_ylabel("Te [eV]")
        ax.set_zlabel("PEC")
        plt.show()


lyman_alpha_lines = {"B": 48.6, "C": 33.7, "N": 24.8, "O": 19.0}

if __name__ == "__main__":
    transitions_list = ["EXCIT", "RECOM"]
    for element, wavelength in lyman_alpha_lines.items():
        for transition in transitions_list:
            PEC(element, wavelength, transition, interp_step=50, plot=True)
