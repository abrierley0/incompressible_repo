Vector Potential-Vorticity Formulation Code
===========================================

*for a three-dimensional cube*
-----------------------------

---------------------------------------------

  For the numerical solution to the flow of a fluid which is incompressible, in a cube where the flow is driven by the motion of the lid,
  
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

3. **Create a Python virtual environment**. Specifically for running this code. (*This environment will ensure that when you install third party Python libraries or 'dependencies' which are required to run the code, that their versions will not conflict with other versions you may have installed elsewhere on your system. Use this environment specifically for running this code. It is a project-specific environment, containing the packages you need to run this code.*)

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

you are now safely 'inside' the environment and can install the packages you need.

Deactivate at any time using,

```bash
deactivate
```

4. **Install required packages**. Using *python package installer* or '*pip*', contain the packages and their versions *within* your new enironment so that they are directly accessible if you have activated the environment,

```bash
pip install matplotlib
pip install pandas
pip install pyyaml
```

###### List of Required Packages

* *matplotlib* - plotting visualisations
* *pandas*
* *pyyaml* - in order to use the configuration file

5. **Run the code**, within your environment,

```bash
python vector_psi_omega.py
```

* Results will be stored in the *results* folder with the filename which you have configured in *config.yaml*. Comparisons with literature are plotted alongside the results. Data extracted from literature are stored in the *lit_data* folder