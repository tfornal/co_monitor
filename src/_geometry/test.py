import os

print(os.path.dirname(os.path.realpath(__file__)))


import pathlib

print(pathlib.Path(__file__).parent.resolve())
