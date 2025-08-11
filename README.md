# SpeAcouPy

**SpeAcouPy** is a Python package for **loudspeaker system modeling** using a lumped-element network approach. 
It can simulate drivers, enclosures, ports, and radiation loads in the electrical, mechanical, and acoustical domains.
You can define components and their parameters in YAML, connect them in arbitrary series/parallel networks, and get system responses such as impedance and SPL.

**Heavy development** — APIs, configuration formats, and features may change at any time without notice.

---

## Installation

If you simply want to use the SpeAcouPy program, install using pipx directly from the GitHub repo:
```
pipx install git+https://github.com/mbrennwa/speacoupy
```
Then run the ``speacoupy`` command.


If you want to use the SpeAcouPy package in your own Python code, clone the GitHub repo:
```
git clone git@github.com:mbrennwa/SpeAcouPy.git SpeAcouPy.git
```
You can also install the ``speacoupy`` program for the local repository:
```
cd SpeAcouPy.git && python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

## System configuration files
SpeAcouPy makes use of YAML configuration files that the describe the loudspeaker system to be modelled. The configuration contains all electrical, mechanical and acoustical elements in the system, and a description (network) of how these elements are linked together. 

ADD DETAILS ON YMAL CONFIGURATION FILES HERE.


## Running a simulation with the ``speacoupy`` progam
To run a simulation according to a given system configuration file using the `speacoupy` program:
```
speacoupy myconfig.yaml
```
Further details:
```
speacoupy --help
```

## License

SpeAcouPy is licensed under the GPL-3.0 — see LICENSE for details.
