# Multi-Agent RL in Sensor Tasking for Space Domain Awareness

## Structure

```
.
├── SSN_RL                              # Core Python Library 
│   ├── agent                           # - Agent classes, algorithms, functions
│   ├── debug                           # - Logger for debugging and counting events
│   ├── environment                     # - Environment classes: Satellite, Sensors, etc.
│   ├── scenarioBuilder                 # - Functions and classes for building scenarios
│   └── utils                            # - Misc utilities
├── data                                # 
│   └── geo_catalog_14Nov2024.txt       # GEO Catalog from 11/14/2024 
└── scripts                             # Test, Training, Plotting, Drafting 
    ├── drafts                          # 
    ├── experiments                     # All Experiments
    │   ├── DQN                         # - DQN training and testing scripts
    │   ├── Heuristic                   # - Heuristic algorithm testing script
    │   ├── Linear                      # - Linear Q-Learning training and testing scripts
    │   ├── QLearning                   # - Q-Table training and testing scripts
    │   ├── QTable                      # - Q-Table first draft
    │   └── plots                       # - Plot generation and test results
    └── tests                           # Tests for Environment
```


## Developer Environment Setup 

1. create conda environment with micromamba 

`micromamba create -n 184-final-project-rl -c conda-forge -y python=3.10`

it used to be

`micromamba create -n 184-final-project -c conda-forge -y python=3.12.4`

2. activate environment 


`micromamba activate 184-final-project-rl`


Note: sometimes I need to run the following command prior to reactivating

`eval "$(micromamba shell hook --shell zsh)"`

3. install dependencies 

`pip install -r requirements.txt`

## running files

### Instructions

Run files from here (.)

python -m scripts.experiments.(Algorithm).(file_name)


### test files

`python -m scripts.tests.basic-maneuver-sim`

`python -m scripts.tests.optical-sensing-test`

`python -m scripts.experiments.DQN.toy1-dqn-train`

### drafts 

`python -m scripts.drafts.driver1`

`python -m scripts.drafts.driver2`


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

