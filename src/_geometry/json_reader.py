"""Reads json file with set of all diagnostic coordinates."""

from pathlib import Path
import json

def read_json_file() -> dict:
    
    
    with open(Path(__file__).parent.resolve() / "coordinates.json") as file:
        data = json.load(file)

    return data 


