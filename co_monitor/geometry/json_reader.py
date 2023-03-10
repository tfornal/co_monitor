"""Reads json file with set of all diagnostic coordinates."""

import json
from pathlib import Path


def read_json_file() -> dict:
    """
    Read a JSON file and return the data as a dictionary.

    Returns:
    -------
    dict :
        Dictionary representation of the data in the JSON file.
    """
    file_path = (
        Path(__file__).parent.parent.parent.resolve()
        / "input_files"
        / "geometry"
        / "coordinates.json"
    )

    with open(file_path) as file:
        data = json.load(file)

    return data


if __name__ == "__main__":
    read_json_file()
