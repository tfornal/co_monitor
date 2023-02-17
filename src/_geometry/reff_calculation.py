__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

from pathlib import Path
from typing import List

import aiohttp
import asyncio
import nest_asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm

from mesh_calculation import CuboidMesh


class ReffVMEC:
    URL = "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0000_+0000/01/00/reff.json?x={}&y={}&z={}"

    def __init__(
        self,
        distance_between_points: int = 10,
        cuboid_size: List[int] = [1350, 800, 250],
        chunk_size: int = 10000,
        save: bool = True,
    ):
        """
        Class to retrieve values of "reff" for a set of Cartesian coordinates using the VMEC code.

        Parameters
        ----------
        distance_between_points : int, optional
            Distance between points of the mesh used to define the plasma domain, by default 10.
        cuboid_size : List[int], optional
            Size of the cuboid containing the plasma domain in millimeters, by default [1350, 800, 250].
        chunk_size : int, optional
            Number of points per chunk, by default 10000.
        save : bool, optional
            If True, saves the result to a file, by default True.
        """

        self.distance_between_points = distance_between_points
        self.cuboid_size = cuboid_size
        self.chunk_size = chunk_size
        self.cuboid_coordinates = (
            self._get_plasma_coordinates() / 1000
        )  # conversion from [mm] to [m]
        self.chunks_nr = self._calc_chunks_nr()
        self.reff_array = self._retrieve_from_API()
        self.reff_df = self.create_df()
        if save:
            self._save_to_file()

    def _get_plasma_coordinates(self) -> np.ndarray:
        """
        Generates an array of cartesian coordinates for a cuboid mesh.

        Returns:
            numpy.ndarray: An array of cartesian coordinates for a cuboid mesh.
        """
        cm = CuboidMesh(self.distance_between_points, self.cuboid_size)
        cuboid_mesh = cm.outer_cube_mesh

        return cuboid_mesh

    def _calc_chunks_nr(self) -> int:
        """
        Calculate the number of chunks required for parallel API requests.

        Returns:
        -------
        int
        The number of chunks required for parallel API requests.
        """
        if len(self.cuboid_coordinates) > self.chunk_size:
            chunks_number = len(self.cuboid_coordinates) // self.chunk_size + 1
        else:
            chunks_number = 1

        print(f"\nNumber of chunks: {chunks_number}")
        print(f"Chunk size: {self.chunk_size}")

        return chunks_number

    def _retrieve_from_API(self):
        """
        Retrieves values of Reff corresponding to the cartesian coordinates from a VMEC API.

        Returns
        -------
        np.ndarray
            Array of Reff values for each plasma point in the cartesian coordinates.
        """

        reff = []
        for chunk in tqdm(range(self.chunks_nr)):

            def get_tasks(session) -> list:
                """
                Generates HTTP GET requests for the plasma points in the current chunk.

                Parameters
                ----------
                session : aiohttp.ClientSession
                    Session object used to create the GET requests.

                Returns
                -------
                List
                    List of GET requests for the plasma points in the current chunk.
                """
                tasks = []
                for plas_point in self.cuboid_coordinates[
                    chunk * self.chunk_size : (chunk + 1) * self.chunk_size
                ]:
                    x, y, z = plas_point
                    tasks.append(session.get(self.URL.format(x, y, z), ssl=False))
                return tasks

            async def get_reff_points():
                """
                Sends HTTP GET requests to the VMEC API to retrieve Reff values for all plasma points.
                """
                async with aiohttp.ClientSession() as session:

                    tasks = get_tasks(session)
                    responses = await asyncio.gather(*tasks)
                    for response in responses:
                        resp_json = await response.json()
                        reff_loaded = resp_json["reff"][0]
                        reff.append(reff_loaded)

            nest_asyncio.apply()
            asyncio.run(get_reff_points())

        return np.array(reff)

    def create_df(self) -> pd.DataFrame:
        """Create a pandas DataFrame with rounded cuboid coordinates and Reff values.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the following columns:
            - "idx": An index of length len(cuboid_coordinates).
            - "x [m]": Rounded cuboid coordinates along the x-axis.
            - "y [m]": Rounded cuboid coordinates along the y-axis.
            - "z [m]": Rounded cuboid coordinates along the z-axis.
            - "Reff [m]": Rounded values of self.reff_array.
        """
        # Round the cuboid coordinates to 4 decimal places.
        rounded_cuboid_coordinates = np.round(self.cuboid_coordinates, decimals=4)

        # Create a DataFrame with the rounded coordinates and Reff values.
        df = pd.DataFrame(
            {
                "idx": np.arange(len(self.cuboid_coordinates)),
                "x [m]": rounded_cuboid_coordinates[:, 0],
                "y [m]": rounded_cuboid_coordinates[:, 1],
                "z [m]": rounded_cuboid_coordinates[:, 2],
                "Reff [m]": np.round(self.reff_array.astype(float), decimals=3),
            }
        )

        return df

    def _save_to_file(self):
        """Save the Reff DataFrame to a file."""

        path = Path(__file__).parent.parent.resolve() / "_Input_files" / "Reff"
        self.reff_df.to_csv(
            path / f"Reff_coordinates-{self.distance_between_points}_mm.dat",
            sep=";",
            index=False,
            na_rep="NaN",
        )


if __name__ == "__main__":
    for precision in [30, 25, 20, 15, 10]:
        ReffVMEC(precision, cuboid_size=[1350, 800, 250], save=True)
