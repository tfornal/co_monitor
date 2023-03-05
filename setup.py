from setuptools import setup, find_packages

# setup(
#     name="co_monitor",
#     version="1.0",
#     description="code for simulation of the signals of the C/O monitor system for Wendelstein 7-X",
#     author="Tomasz Fornal",
#     author_email="tomasz.fornal6@gmail.com",
#     packages=["co_monitor"],
#     # install_requires=requirements.txt,
#     classifiers=[
#         "Development Status :: 1 - Testing",
#         "Intended Audience :: Science/Research",
#         "Operating System :: POSIX :: Linux",
#         "Programming Language :: Python :: 3.10",
#     ],
# )


setup(
    name="co_monitor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
