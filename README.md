# B-AMA
B-AMA (Basic dAta-driven Models for All) is an easy, flexible, and fully coded Python-written protocol for the application of data-driven models (DDM) in hydrological problems. <br />
Regardless of the type of input data and the case-study under investigation, B-AMA consists of the following straightforward steps: <br />
1 - Installation: Install all the required pre-requisites libraries (see dependencies.txt for a complete list). <br />
2 - loading: Load the input data in the input//case_study folder. Make sure that the dependent variable is in the last column of your csv file. <br />
3 - Define settings in the configuration file. 
The information to be specified in the default configuration file are the case study name, the periodicity of the variable to be forecasted, the initial year and the final year with observations, and the modelling techniques to be used. <br />
4 - Run the protocol: Open the Anaconda prompt terminal and navigate to the B-AMA folder. Type ‘python ddm_run.py’ in the terminal window. Alternatively, it is possible to run directly the ddm_run.py script from Spyder. <br />
B-AMA will then run all the protocol steps.

