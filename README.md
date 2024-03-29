# B-AMA

B-AMA (Basic dAta-driven Models for All) is an easy, flexible, and fully coded Python protocol for applying data-driven models (DDM) in hydrological problems.

## Overview

B-AMA simplifies the process of applying data-driven models to hydrological problems, regardless of the input data type or case study. The protocol consists of the following straightforward steps:

1. **Installation**: Start by installing all the required libraries using the provided `environment.yml` file. Run the following command in your terminal:
```
conda env create -f environment.yml
conda activate b_ama_envir
```
This will create a Conda environment and activate it.

2. **Loading Data**: Place your input data in the `input/case_study` folder. Ensure that the dependent variable is in the last column of your CSV file.

3. **Define Settings**: Specify the settings in the configuration file. Required information includes the case study name, the periodicity of the variable to be forecasted, the initial and final years with observations, and the modeling techniques to be used.

4. **Run the Protocol**: Open the Anaconda Prompt terminal, navigate to the B-AMA folder, and run the following command:
```
python ddm_run.py
```
Alternatively, you can run the `ddm_run.py` script directly from Spyder.

## Additional Information
- **Manuscript**: You can find more information about the methodology, validation, and applications of the software in the following publication:
  - [B-AMA: A Python-coded protocol to enhance the application of data-driven models in hydrology](https://www.sciencedirect.com/science/article/pii/S1364815222003097)
- **Input Data Requirements**: Ensure that your input data is formatted correctly and meets the requirements specified in the documentation.
- **Configuration Settings**: Consult the documentation for guidance on configuring the settings in the configuration file.
- **Examples and Use Cases**: Explore examples and use cases to see how B-AMA can be applied in various scenarios.
- **Contributing and Support**: We welcome contributions from the community. If you encounter any issues or have suggestions for improvement, please open an issue on GitHub.

## License

This project is licensed under the [GNU General Public License version 3](LICENSE).
