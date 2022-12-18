import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io, fnmatch
from pathlib import Path
from collections import namedtuple


class PEC:
    def __init__(
        self,
        element,
        wavelength,
        transitions,
        ne_samples_amount=20,
        Te_samples_amount=20,
    ):
        self.element = element
        self.wavelength = wavelength
        self.ne_samples_amount = ne_samples_amount
        self.Te_samples_amount = Te_samples_amount
        self.transitions = transitions
        self.pec_file_path = self.select_pec_file()
        self.total_pec_file = self.separate_pec_file()
        self.comments_df = self.read_pec_comments()
        self.transitions_list = self.select_radiation_type()
        self.mesh_nodes = self.pec_mesh_nodes()
        self.interp_df_shape = self.calc_interp_df_shape()
        self.analyse_pec()

    def select_pec_file(self):
        pec_element_path = Path.cwd() / "_Input_files" / "PEC" / self.element
        pec_file_path = list(Path(pec_element_path).glob("*.dat"))[0]

        return pec_file_path

    def separate_pec_file(self):
        """
        TODO dafaq?
        """

        num_list = []
        with open(self.pec_file_path, "r") as fh:
            for line in fh:
                num_list.append(line)

        dumped_array = np.asarray(num_list)
        new_array = []

        for i in range(len(dumped_array)):
            z = dumped_array[i].replace("\n", "")
            new_array = np.append(new_array, z)

        elements_in_row = []
        total_pec_file = []
        for i in range(len(new_array)):
            new_array[i] = new_array[i].replace("    ", " ")
            new_array[i] = new_array[i].replace("   ", " ")
            new_array[i] = new_array[i].replace(" A", "A")
            new_array[i] = new_array[i].replace("+T", " +T")
            new_array[i] = new_array[i].replace(" = ", "=")
            new_array[i] = new_array[i].replace(" =", "=")
            new_array[i] = new_array[i].replace("/", " ")
            x = new_array[i].split(" ")

            while "" in x:
                x.remove("")
            for i in range(len(x)):
                total_pec_file.append(x[i])
            elements_in_row = np.append(elements_in_row, len(x))

        return total_pec_file

    def read_pec_comments(self):
        """Czyta wszystkie informacje zawarte w tabeli z komentarzami"""

        start_index = 0
        stop_index = 0
        comments = []

        with open(self.pec_file_path, "r") as file:
            file = file.readlines()
            file_length = len(file)
            for idx in range(file_length):
                if (
                    "C  ISEL  WAVELENGTH      TRANSITION            TYPE    METASTABLE  IMET NMET  IP\n"
                    in file[idx]
                ):
                    start_index = idx
            for idx in range(start_index, file_length):
                if file[idx] == "C\n":
                    stop_index = idx
                    break

        for idx in range(start_index, stop_index):
            x = file[idx].split("\n")
            while "" in x:
                x.remove("")
            comments = np.append(comments, x)

        tekst = "\n".join(comments)
        comments_df = pd.read_fwf(io.StringIO(tekst)).drop("C", axis=1).drop(0)
        comments_df = comments_df.astype({"ISEL": float, "WAVELENGTH": float})

        return comments_df

    def select_radiation_type(self):
        transmission_type = {}
        for i in range(1, len(self.comments_df)):
            if self.comments_df.loc[i, "WAVELENGTH"] == self.wavelength:
                if len(self.transitions) != 0:
                    for trans in self.transitions:
                        if trans == self.comments_df.loc[i, "TYPE"]:
                            transmission_type[
                                int(self.comments_df.loc[i, "ISEL"])
                            ] = self.comments_df.loc[i, "TYPE"]
                else:
                    transmission_type[
                        int(self.comments_df.loc[i, "ISEL"])
                    ] = self.comments_df.loc[i, "TYPE"]

        return transmission_type

    def pec_mesh_nodes(self):
        Ne, Te, first_pec_idx = [], [], []
        for transition_type_comment_number in self.transitions_list:
            for i in range(len(self.total_pec_file)):
                if self.total_pec_file[i] == f"{self.wavelength}A":
                    if (
                        self.total_pec_file[i + 6]
                        == f"ISEL={transition_type_comment_number}"
                    ):
                        ne_probes_number = int(
                            self.total_pec_file[i + 1]
                        )  # ne samples probe number
                        te_probes_number = int(
                            self.total_pec_file[i + 2]
                        )  # Te samples probe number
                        ne_idx = int(i + 7)  # index pierwszej wartosci ne x Te

                        first_pec_idx.append(
                            int(ne_idx + ne_probes_number + te_probes_number)
                        )  #  wpisuje jaki index ma konkretna
                        Ne.append(
                            self.total_pec_file[(ne_idx) : (ne_idx + ne_probes_number)]
                        )
                        Te.append(
                            self.total_pec_file[
                                (ne_idx + ne_probes_number) : (
                                    ne_idx + ne_probes_number + te_probes_number
                                )
                            ]
                        )

        total = namedtuple("Transition", "idx ne Te")
        templist = list(zip(first_pec_idx, Ne, Te))
        mesh_nodes_values = [total(*x) for x in templist]
        transitions_keys = {el: 0 for el in self.transitions_list.values()}
        mesh_nodes = dict(zip(transitions_keys, mesh_nodes_values))

        return mesh_nodes

    def calc_interp_df_shape(self):
        shape_array = np.zeros([len(self.mesh_nodes), 5]).astype(int)
        for i, j in enumerate(self.mesh_nodes):
            shape_array[i, 0] = len(self.mesh_nodes[j].ne)
            shape_array[i, 1] = len(self.mesh_nodes[j].Te)
            ne_probes_number = int(shape_array[i, 0])
            te_probes_number = int(shape_array[i, 1])
            T_e = self.Te_samples_amount * (te_probes_number - 1)
            n_e = self.ne_samples_amount * (ne_probes_number - 1)
            shape_array[i, 2:5] = (n_e, T_e, 3)

        transitions = namedtuple(
            "Transition", "ne Te ne_interp Te_interp another_dimension"
        )
        shape_info = [transitions(*i) for i in shape_array]
        transitions_keys = {el: 0 for el in self.transitions_list.values()}
        interp_df_shape = dict(zip(transitions_keys, shape_info))

        return interp_df_shape

    def pec_interp_2d(self, pec_nodes_array, transition):
        # 1st dimension interpolation
        interp_te_pec = np.zeros(
            [
                len(pec_nodes_array),
                self.Te_samples_amount * (len(pec_nodes_array[0]) - 1),
                3,
            ]
        )

        for ne_value in range(len(pec_nodes_array)):
            te_raw_nodes = pec_nodes_array[ne_value, :, 1]
            pec_raw_values = pec_nodes_array[ne_value, :, 2]

            Te_interpolated_all = []
            pec_interpolated = []
            for Te_dimension in range(len(pec_nodes_array[ne_value]) - 1):
                Te_interp_two_points = np.linspace(
                    te_raw_nodes[Te_dimension],
                    te_raw_nodes[Te_dimension + 1],
                    self.Te_samples_amount,
                    endpoint=False,
                )
                PEC = np.linspace(
                    pec_raw_values[Te_dimension],
                    pec_raw_values[Te_dimension + 1],
                    self.Te_samples_amount,
                    endpoint=False,
                )
                Te_interpolated_all = np.append(
                    Te_interpolated_all, Te_interp_two_points
                )
                pec_interpolated = np.append(pec_interpolated, PEC)

            interp_te_pec[ne_value, :, 0] = pec_nodes_array[ne_value, 0, 0]
            interp_te_pec[ne_value, :, 1] = Te_interpolated_all
            interp_te_pec[ne_value, :, 2] = pec_interpolated

        # 2nd dimension interpolation
        interp_ne_te_pec = np.zeros(
            [
                self.ne_samples_amount * (len(pec_nodes_array) - 1),
                len(interp_te_pec[0]),
                3,
            ]
        )

        for te_value in range(len(interp_te_pec[0, :])):
            ne = interp_te_pec[:, te_value, 0]
            te = interp_te_pec[:, te_value, 1]
            pec = interp_te_pec[:, te_value, 2]

            ne_interpolated_all = []
            Te_interpolated_all = []
            pec_interpolated_all = []
            for j in range(len(interp_te_pec) - 1):
                ne_interpolated_between_two_points = np.linspace(
                    ne[j], ne[j + 1], self.ne_samples_amount, endpoint=False
                )
                Te_interpolated_between_two_points = np.linspace(
                    te[j], te[j + 1], self.ne_samples_amount, endpoint=False
                )
                pec_interpolated_between_two_points = np.linspace(
                    pec[j], pec[j + 1], self.ne_samples_amount, endpoint=False
                )
                # print(nee)
                ne_interpolated_all = np.append(
                    ne_interpolated_all, ne_interpolated_between_two_points
                )
                Te_interpolated_all = np.append(
                    Te_interpolated_all, Te_interpolated_between_two_points
                )
                pec_interpolated_all = np.append(
                    pec_interpolated_all, pec_interpolated_between_two_points
                )

            interp_ne_te_pec[:, te_value, 0] = ne_interpolated_all
            interp_ne_te_pec[:, te_value, 1] = Te_interpolated_all
            interp_ne_te_pec[:, te_value, 2] = pec_interpolated_all

        def plotter():

            X = interp_ne_te_pec[:, :, 0]
            Y = interp_ne_te_pec[:, :, 1]
            Z = interp_ne_te_pec[:, :, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            # ax.plot_surface(X, Y, Z,cmap = "plasma", rstride = 1)
            ax.plot_wireframe(X, Y, Z, rstride=1)
            plt.title(f"PEC - {transition}")
            ax.set_xlabel("Ne [1/cm3]")
            ax.set_ylabel("Te [eV]")
            ax.set_zlabel("PEC")

        if __name__ == "__main__":
            plotter()

        return interp_ne_te_pec

    def analyse_pec(self):
        def calculator(transition):
            ne_probes_number = self.interp_df_shape[transition].ne
            te_probes_number = self.interp_df_shape[transition].Te
            total_pec_amount = ne_probes_number * te_probes_number
            pec_nodes_array = np.zeros([total_pec_amount, 3])
            subsequent_idx = 0
            first_pec_idx = self.mesh_nodes[transition].idx
            n_e = self.mesh_nodes[transition].ne
            T_e = self.mesh_nodes[transition].Te
            for i in n_e:
                for j in T_e:
                    pec_nodes_array[subsequent_idx][0] = i
                    pec_nodes_array[subsequent_idx][1] = j
                    pec_nodes_array[subsequent_idx][2] = self.total_pec_file[
                        first_pec_idx + subsequent_idx
                    ]
                    subsequent_idx += 1
            pec_nodes_array = pec_nodes_array.reshape(
                ne_probes_number, te_probes_number, 3
            )
            interp_ne_pec = self.pec_interp_2d(pec_nodes_array, transition)

            return interp_ne_pec

        ne_interp = self.interp_df_shape[self.transitions[0]].ne_interp
        Te_interp = self.interp_df_shape[self.transitions[0]].Te_interp
        another_dimension = self.interp_df_shape[self.transitions[0]].another_dimension
        transitions_number = len(self.interp_df_shape)

        interpolated_pec_df = np.zeros(
            [transitions_number, ne_interp, Te_interp, another_dimension]
        )
        for iterator, transition in enumerate(self.interp_df_shape):
            interp_ne_pec = calculator(transition)
            interpolated_pec_df[iterator] = interp_ne_pec
        return interpolated_pec_df, self.transitions_list


### correct the function calling, to namedtuple
if __name__ == "__main__":
    transitions = ["EXCIT", "RECOM"]
    pec = PEC("C", 33.7, transitions)
    # pec = PEC("B", 48.6, transitions)
    # pec = PEC("O", 19.0, transitions)
    # pec = PEC("N", 24.8, transitions)
