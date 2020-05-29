Structured Anomalies
=======================

Setup
------------------------
The setup process for this repository requires the following steps:

### Download
Download the repository.

    git clone https://github.com/raphael-group/structured-anomalies.git
    
    
### Input

Finding an anomaly requires at least one input file. 

##### Gene-to-score file
This is a `.tsv` file. It is needed for all anomalies except the submatrix anomaly. 

Each line in this file associates a node with a score:

    A    -1
    B    2.5
    C    3
    
##### HDF5 file
This is a file needed only for the submatrix anomaly. It contains a square matrix of scores. An example of how to create one of these is in `write_example_anomalies.py`.
    
##### Edge list file
This is a `.tsv` file. It is needed only for the cutsize and connected anomalies. Each edge in this file corresponds to an edge in the network.

    A    C
    B    C
    
### Output
For most anomalies, `find_anomaly.py` writes to the specified output file a set of anomalous nodes corresponding to whatever structure was inputted. Each line in the output file is a node:

    B
    C

For the submatrix anomaly, `find_anomaly.py` writes a list of rows and a list of columns corresponding to the submatrix.

Additional information
----------------

### Examples
See the `examples` directory for an example of each anomaly type that should complete very quickly on most machines.

