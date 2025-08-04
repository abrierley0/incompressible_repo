Vector Potential-Vorticity Formulation Code
===========================================

*for a flow where an arbitrary fluid is contained within an impermeable cube, and fluid motion is driven by the motion of the top surface*
-----------------------------

---------------------------------------------

The vector potential-vorticity formulation is a three dimensional extension of the streamfunction-vorticity formulation. The vector potential has been known to exist for a while, but storage and computational limitations prevented its wide use, also a difficulty with the vector potential boundary conditions. Hirasaki and Hellums found a general boundary conditons for this formulation, which works well if the walls are impermeable. Through-flows are more difficult.

This code is written for a three-dimensional cube with a moving top wall.

#### Contents

* _**lit_data**_ - figures from the reference data in .png format, and also extracted data from these figures in .csv format (extraction done using engauge digitizer)
* _**results**_ - storage for results, including slice visualisations in .png format and centreline velocity data in .csv format
* _**config.yaml**_ - configuration file to configure the simulation as you wish
* _**vector_psi_omega.py**_ - the code

#### Running the Code
  
1. **Configure your parameters** as required for simulation in the *config.yaml* file, by typing the following in the Linux terminal,

```bash
code config.yaml
```

2. **Install Python**. If you are using the Ubuntu linux distribution, the package manager to install Python is,

```bash
sudo apt install python3
```

and double check the version,

```bash
python3 --version
```

3. **Create a Python virtual environment**. To contain packages needed specifically to run this code.

Create the Python virtual environment by running,

```bash
python -m venv myenv
```

or,

```bash
python3 -m venv myenv
```

*myenv* can be any name of your choosing. This will contain your packages.

Then activate the environment,

```bash
source myenv/bin/activate
```

you are now safely 'inside' the environment and can install the packages you need without causing conflict if a different package version is installed elsewhere on the system.

Deactivate at any time using,

```bash
deactivate
```

4. **Install packages required to run the code**. Using *python package installer* or '*pip*', contain the packages and their versions *within* your new enironment so that they are directly accessible if you have activated the environment,

```bash
pip install matplotlib
pip install pandas
pip install pyyaml
```

###### List of Required Packages

* *matplotlib* - plotting visualisations
* *pandas* - used to read the .csv data from the literature and simulation for plotting purposes
* *pyyaml* - in order to use the configuration file

5. **Run the code**, within your environment,

```bash
python vector_psi_omega.py
```

* Results will be stored in the *results* folder with the filename which you have configured in *config.yaml*. Comparisons with literature are plotted alongside the results. Data extracted from literature are stored in the *lit_data* folder