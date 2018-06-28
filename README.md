# Peakaboo

Peakaboo is a software package for analysis of transient absorption (TA) data. It can identify self-consistent spectrally and temporally evolvin signatures of charge carriers after photoexcitation in TA data. With minimal assumption, our algorithm recovers and visualizes the spectral and kinetics information of the individual population by combining methods such as multivariate adaptive regression spline fitting and data clustering.



## Software Requirements

- Required softwares are listed in requirements.txt
- ```pip install -r requirements.txt```
- ```pip``` will check the softwares above except ```py-earth```



## Installation instruction

- Install from pip:
    
    ```pip install PEAKABOO```


## How to use the package

- ```import peakaboo```

### 1. Running __main__.py
### This is a user interactive function that inputs user data and outputs peak characters

- User provides data in .csv or .txt format and specifies wavelength range and time-zero
- Reduce noise in data
- Find peaks in data with user-specific selection criteria
- Classify peak outputs
- Visualize and fit peak kinetics to exponential function
- Peak info is saved

### 2. Running individual function
- The following ``.py`` files contain individual function for each step above: 
    - data_smoothing.py: load data, reduce noise in data
    - smoothing_visualize.py: visualize data in 2D contour plot
    - find_peaks.py: find peaks
    - peak_character.py: identify peak_utils
    - peak_classify.py: classify disorganized peak info
    - kinetics_fitting.py: fit peak dynamics to mono-exponential kinetics
for example: peakaboo.data_smoothing


## License

MIT