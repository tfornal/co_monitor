import numpy as np

import time
from mesh_calculation_OOP import PlasmaMesh
from tqdm import tqdm
import asyncio
import nest_asyncio
import aiohttp


class ReaderReffVMEC:
    """
    """
    def __init__(
        self,
        distance_between_points: int = 250,
        cuboid_size: np.array = [2000, 2000, 2000],
        savetxt: bool = False,
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
        self.savetxt = savetxt
        self.distance_between_points = distance_between_points
        self.xyz_coordinates = (
            PlasmaMesh(distance_between_points, cuboid_size).outer_cube_mesh / 1000
        )  # conversion from mm to m
        self.xyz_coordinates = self.xyz_coordinates[:]
        self.reff_array = self.read_reff_calculated()
        if savetxt:
            self.savetxt()
    def read_reff_calculated(self):

        """
        Reads values of Reff corresponding to the cartesian coordinates. VMEC
        accepts coordinates in meters.
        """
        plasma_coordinates = self.xyz_coordinates
        url = "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0000_+0000/01/00/reff.json?x={}&y={}&z={}"

        ##### async
        results = []

        chunk_size = (
            10000  ###if chunksize = 0 > chunks number == 1 chunk_size >0 warunek
        )

        def get_coord_chunks():
            #            if chunk_size == 0:
            #                chunks_number = 1
            if len(plasma_coordinates) > chunk_size:
                chunks_number = len(plasma_coordinates) // chunk_size + 1
            else:
                chunks_number = 1
            return chunks_number

        chunks_nr = get_coord_chunks()
        print(f"Number of chunks: {chunks_nr}")
        print(f"Chunk size: {chunk_size}")

        for chunk in tqdm(range(chunks_nr)):

            def get_tasks(session):
                tasks = []
                for plas_point in plasma_coordinates[
                    chunk * chunk_size : (chunk + 1) * chunk_size
                ]:
                    x, y, z = plas_point
                    tasks.append(session.get(url.format(x, y, z), ssl=False))
                return tasks

            async def get_reff_points():
                async with aiohttp.ClientSession() as session:

                    tasks = get_tasks(session)
                    responses = await asyncio.gather(*tasks)
                    for response in responses:
                        resp_json = await response.json()
                        reff = resp_json["reff"][0]
                        results.append(reff)

            nest_asyncio.apply()
            asyncio.run(get_reff_points())

            np.savetxt(
                f"C:/Users/tofo/Desktop/reff_calculation_OOP/reff/Reff_VMEC_reff-{chunk}.txt",
                np.array(results, dtype=float),
                fmt="%.3e",
            )
            results = []

        reff_array = np.zeros([len(plasma_coordinates), 4])
        reff_array[:, 0] = np.arange(len(plasma_coordinates))
        reff_array[:, 1:4] = plasma_coordinates[:, :3]

        np.savetxt("Plasma_coords.txt", reff_array, newline="\n", fmt="%.3e")

    def save_reff(self, savetxt):
        """
        Saving data to *.txt file.
        """
        with open(
            "Reff_coordinates-{}mm.txt".format(self.distance_between_points), "a"
        ) as file:
            np.savetxt(
                f"Reff_coordinates-{self.distance_between_points}mm.txt",
                self.reff_array,
                header=f"Coordinates X[mm], Y[mm], Z[mm], Reff, Intensity; \nSingle block size: {self.distance_between_points} x {self.distance_between_points} mm",
            )
        print("Reff successfully saved!")



t1 = time.time()
if __name__ == "__main__":
    ReaderReffVMEC(100, cuboid_size=[2000, 2000, 2000], savetxt=True)
t2 = time.time()

print(f"\n Successfully finished within {t2-t1} seconds.")
