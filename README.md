

# The C/O monitor code for signal's simulations

## --- The code is currently under intensive development ---

The code requires Python v.> 3.9.
The code was created to perform various calculations of potential C/O monitor signals and validate experimental data of the Lyman-alpha lines measurement of hydrogen like ions of B, C, N and O. The code calculates only the line intensities of the given transitions but does not calculate the background radiation (at least not for now). It includes all the obstacles such as W7-X Port, ECRH shields, collimators, curvatures of dispersive elements and position od detectors. 

TBC


## The code consists of three main sections:
### Module 1 - Geometry calculation

The section calculates the observed plasma volume by the respective energy channel (possible options - boron (B), carbon (C), nitrogen (N) and oxygen (O)). 
directory - co_monitor/geometry 

First - by the use of VMEC code (external API) the respective plasma volume in a defined vicinity is calculated. For each plasma point defined in carthesian coordinate system (x,y,z  mm) the Reff (effective minor radius) value is calculated. 
Next, the code simulates geometry of the C/O monitor system and calculates how much radiation reaches the respective detector. It performes ray-tracing-like operation in order to obtain results wit the highest precision. 
Calculations are preformed by : 
#### co_monitor/geometry/simulation.py

Visualization (generated by the use of co_monitor/src/geometry/visualization.py module) of simulated environment (C/O monitor system and W7-X) is presented below :

![image](https://user-images.githubusercontent.com/53053987/218479276-a48353f5-eb2e-41cd-b642-afe016b43731.png)

TBC


### Module 2 - Emissivity calculation

directory - co_monitor/emissivity/reader.py
Based on the geometry file calculated with the use of Module 1 - the code calculates the total photon intensity flux reaching the respective detector.
It also provieds information with radial emissivity distribution for two transition types - excitation and recombination. CX processes are planned to be included in the future.

TBC
