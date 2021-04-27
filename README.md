# WolfSim

Wolf Simulator for CSE 6730, Georgia Institute of Technology, Spring 2021

![Wolfsim Logo](bin/WolfSim_Logo.png?raw=true "WolfSim Logo")

## Overview
Our team has been tasked by the Alaska Department of Fish and Game, working in conjunction with the Alaska Department of Transportation and the Anchorage Port Authority, to assess various tracking options for wolf populations inside an area of interest surrounding Anchorage, including the majority of the Alaska Range and segments of the Wrangell and Chugach Mountains.  More specifically, this area spans a rectangular area 200 miles north, east, and west of the city of Anchorage, as well as 100 miles to the South. The main objective is to assess wolf-tracking methods under varying environmental factors and communications constraints to better assist in these conservation efforts. A simulation environment is created in order to evaluate the current tracking methods that use radio collars, satellite collars, or aerial spotting.   Each method has its advantages and disadvantages that should be taken into consideration such as range, error, and consistency. 

## Required Packages
The core Python packages are listed below, as well as in the setup.py file. Python 3.7 was used in development, but any Python 3.x is expected to work correctly.
* Mesa
* Numpy
* Matplotlib
* Scipy
* Networkx


## Files
* *WolfsimModel*
    * The framework for the simulation is defined here. The mesa model is used to define the environment in the form of a grid, 
    the scheduling of agents in the environment, the initial placement of those agents, the events at each time step of the simulation, 
    as well as a few other supplimentary functions.
* *WolfsimAgents*
    * The definition and behavior of the wolf agent with movement, age parameters, and health indicators.
    The mesa agent class is used to build the wolf agent. The movement of the wolf used individual behaviors to
    model the swarm-like movement of the pack.
* *RunWolfsim*
    * The functions for running single cases, or batch cases of the simulation.
     Functions are also available for storing data in the simulation using the mesa datacollector class.
* *utils*
    * Additional functions needed throughout the other three modules.
## Data
* Elevation Files
    * The file defining the elevation in feet of the area of interest.
* Vegetation Files
    * The file defining the different types of vegetation in the environment in the same region as the elevation data. 
    The data is given a label 2 for vegetation and 0 for no vegetation.
    
    The data was pulled from online sources in the area of Alaska surrounding Anchorage.
 
## Developers
Caleb Harris - caleb.harris94@gatech.edu    
Johnie Sublett - jsublett3@gatech.edu   
Group #16