## Arlib 

Arlib is toolkit for automated reasoning.

## Milestones

- Make setup.py work (release the repo as a Python library )

## Setup a Local Development Environment

~~~~
bash setup_local_env.sh
~~~~

The script will:
- Create a Python virtual environment if it doesn't exist
- Activate the virtual environment and install dependencies from requirements.txt
- Download required solver binaries (cvc5, mathsat, z3)
- Run unit tests if available

TBD:
- Test the scripts for different platforms, editors, etc.

## Usage


~~~~
arlib/
├── arlib/           # Main library code
├── benchmarks/      # Benchmark files and test cases
├── bin_solvers/     # Binary solver executables
├── docs/            # Documentation files
├── examples/        # Utility scripts
├── setup.py         # Package setup configuration
├── pytest.ini       # PyTest configuration
└── requirements.txt # Project dependencies
~~~~

## Research Topics

For Summer Research, Final Year Project Topics, please refer to
`docs/topics.rst`

## Contributing
Contributions are welcome. Please refer to the repository for detailed instructions on how to contribute. 


## Documentation
We release the docs here:
https://pyarlib.readthedocs.io/en/latest/


## Contributors

Primary contributors to this project:
- rainoftime / cutelimination
- JasonJ2021
- ZelinMa557 
- Harrywwq
- little-d1d1
