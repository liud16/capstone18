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

- Run package from command line:

    ``python -m peakaboo``
### Workflow:

    - User provides data in .csv or .txt format and specifies wavelength range and time-zero
    - Reduce noise in data
    - Find peaks in data with user-specific selection criteria
    - Classify peak outputs
    - Visualize and fit peak kinetics to exponential function
    - Peak info is saved



## License

MIT