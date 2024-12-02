# Multi-Agent RL in Sensor Tasking for Space Domain Awareness

## Developer Environment Setup 

1. create conda environment with micromamba 

`micromamba create -n 184-final-project -c conda-forge -y python=3.12.4`

2. activate environment 


`micromamba activate 184-final-project`


Note: sometimes I need to run the following command prior to reactivating

`eval "$(micromamba shell hook --shell zsh)"`

3. install dependencies 

`pip install -r requirements.txt`

## running files


`python -m tests.basic-maneuver-sim.py`

`python -m tests.optical-sensing-test.py`


## Library Documentation 

[skyfield](https://rhodesmill.org/skyfield/toc.html)

[skyfield: earth satellites](https://rhodesmill.org/skyfield/earth-satellites.html)

## Astrodynamics Checklist

### check the following capabilities
- [x] load states from file or string
- [x] have ground observer
- [x] compute visibility from ground to satellite
- [x] do propagation to create ephemerides 
- [x] convert states to pos/vel
- [x] make modifications to states after loading (maneuver simulation)

### generate the following scripts 
- [x] GEO visibility opportunity plot 
- [x] Impulsive maneuver simulation 
- [x] Covariance growth plot simulation

## Data Documentation 


[space-track.org (requires an account)](https://www.space-track.org/#/Landing)

[current GEO catalog from space-track](https://www.space-track.org/basicspacedata/query/class/gp/EPOCH/%3Enow-30/MEAN_MOTION/0.99--1.01/ECCENTRICITY/%3C0.01/OBJECT_TYPE/payload/orderby/NORAD_CAT_ID,EPOCH/format/3le)

