__author__ = "T. Fornal"
__email__ = "tomasz.fornal6@gmail.com"

import aiohttp
import asyncio
import nest_asyncio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from mesh_calculation import CuboidMesh
import pandas as pd


class ReffVMEC:
    URL = "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0000_+0000/01/00/reff.json?x={}&y={}&z={}"

    def __init__(
        self,
        distance_between_points: int = 10,
        cuboid_size: list = [1350, 800, 250],
        chunk_size: int = 10000,
        save: bool = True,
    ):
        """Odczytywanie wspolrzedne xyz uzyskanych przez plik "mesh_calculation.py"
        oraz obliczanie przez kod VMEC reff i zpaisywanie do pliku.
        Uruchamiane tylko na serwerach greifswaldowskich. Pobiera bezposrednio
        tablice z funkcji "mesh_calculation" i wykonuje obliczenia.

        Args:
            distance_between_points (int): _description_. Defaults to 250.
            cuboid_size (np.array): _description_. Defaults to [2000, 2000, 2000].
            savetxt (bool): _description_. Defaults to False.
        """
        self.distance_between_points = distance_between_points
        self.cuboid_size = cuboid_size
        self.chunk_size = chunk_size

        self.cuboid_coordinates = self._get_plasma_coordinates() / 1000
        self.reff_df = self.read_reff_calculated()
        if save:
            self._save_to_file()

    def _get_plasma_coordinates(self):
        cm = CuboidMesh(self.distance_between_points, self.cuboid_size)
        cuboid_mesh = cm.outer_cube_mesh

        return cuboid_mesh

    def read_reff_calculated(self):

        """
        Reads values of Reff corresponding to the cartesian coordinates. VMEC
        accepts coordinates in meters.
        """
        reff = []

        # async
        def calc_chunks_nr():
            #            if chunk_size == 0:
            #                chunks_number = 1
            if len(self.cuboid_coordinates) > self.chunk_size:
                chunks_number = len(self.cuboid_coordinates) // self.chunk_size + 1
            else:
                chunks_number = 1
            return chunks_number

        chunks_nr = calc_chunks_nr()
        print(f"\nNumber of chunks: {chunks_nr}")
        print(f"Chunk size: {self.chunk_size}")

        for chunk in tqdm(range(chunks_nr)):

            def get_tasks(session):
                tasks = []
                print(self.cuboid_coordinates)
                for plas_point in self.cuboid_coordinates[
                    chunk * self.chunk_size : (chunk + 1) * self.chunk_size
                ]:
                    x, y, z = plas_point
                    tasks.append(session.get(self.URL.format(x, y, z), ssl=False))
                return tasks

            async def get_reff_points():
                async with aiohttp.ClientSession() as session:

                    tasks = get_tasks(session)
                    responses = await asyncio.gather(*tasks)
                    for response in responses:
                        resp_json = await response.json()
                        reff_loaded = resp_json["reff"][0]
                        reff.append(reff_loaded)

            nest_asyncio.apply()
            asyncio.run(get_reff_points())

        reff_array = np.zeros([len(self.cuboid_coordinates), 3])
        reff_array = self.cuboid_coordinates[:, :3].round(4)

        df = pd.DataFrame()
        df["idx"] = np.arange(len(self.cuboid_coordinates))
        df["x [m]"] = reff_array[:, 0]
        df["y [m]"] = reff_array[:, 1]
        df["z [m]"] = reff_array[:, 2]
        df["Reff [m]"] = np.array(reff).astype(float).round(3)
        print(df)
        return df

    def _save_to_file(self):
        path = Path(__file__).parent.parent.resolve() / "_Input_files" / "Reff"
        self.reff_df.to_csv(
            path / f"Reff_coordinates-{self.distance_between_points}_mm.dat",
            sep=";",
            index=False,
            na_rep="NaN",
        )


if __name__ == "__main__":
    for precision in [30, 25, 20, 15, 10]:
        try:
            ReffVMEC(precision, cuboid_size=[1350, 800, 250], save=True)
        except:
            pass
