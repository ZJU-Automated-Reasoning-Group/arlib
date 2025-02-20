# Set up the working environment without Dockerfile
# To be tested
#!/bin/bash

# Exit on any error
set -e  # should we do this?

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "Setting up arlib environment..."

# 1. Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists."
fi

# 2. Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 3. Download solver binaries
echo "Downloading solver binaries..."
python "${SCRIPT_DIR}/bin_solvers/download.py"
# TODO: check where the binary solvers have been downloaded in `bin_solver`, 
# cvc5, mathsat, z3.

# 4. Run tests
echo "Running tests..."
if [ -f "${SCRIPT_DIR}/unit_tests.sh" ]; then
    bash "${SCRIPT_DIR}/unit_tests.sh"
else
    echo "Warning: unit_tests.sh not found"
fi

echo "Setup completed successfully!"