# Peakaboo

Peakaboo is a software package for analysis of transient absorption (TA) data. It can identify self-consistent spectrally and temporally evolvin signatures of charge carriers after photoexcitation in TA data. With minimal assumption, our algorithm recovers and visualizes the spectral and kinetics information of the individual population by combining methods such as multivariate adaptive regression spline fitting and data clustering.



## Software Requirements

- Required softwares are listed in requirements.txt
- ```pip install -r requirements.txt```



## Installation instruction

- Clone git repository to local:
    
    ```git clone https://github.com/liud16/peakaboo.git ```
    
- Run main function in command line:
    
    In ```peakaboo/peakaboo```, run peakaboo.py at command line: ```python peakaboo.py```



## Software overflow

- User provides data in .csv or .txt format and specifies wavelength range and time-zero
- Reduce noise in data
- Find peaks in data with user-specific selection criteria
- Classify peak outputs
- Visualize and fit peak kinetics to exponential function


## License

MIT