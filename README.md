Structured Anomalies
=======================

Setup
------------------------
The setup process for this repository requires the following steps:

### Download
Download the repository.

    git clone https://github.com/raphael-group/structured-anomalies.git
    
   
##### Required

* Linux/Unix
* [Python (2.7 or 3.6)](http://python.org/)
* [NumPy (1.17)](http://www.numpy.org/)
* [SciPy (1.3)](http://www.scipy.org/)
* [h5py (2.10)](http://www.h5py.org/)
* [NetworkX (2.4)](https://networkx.github.io/)
    
### Input

##### Anomaly type

* `connected`: This anomaly is a connected subgraph of some larger graph. To find this type of anomaly, use [NetMix](https://github.com/raphael-group/netmix).
* `cutsize`: This anomaly is a subgraph of some larger graph such that the weight of the cut is less than some predefined rho. 
* `line`: This anomaly is a connected subgraph of a line-shaped graph, which means it's a smaller line.
* `submatrix`: This anomaly is a submatrix of a matrix. 
* `unconstrained`: This anomaly is a subset of vertices of a graph where edges are nonexistent or ignored.

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

