# Set up the working environment without Dockerfile
# FIXME: be tested
#!/bin/bash
# Exit on any error
set -e  # should we do this?

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "Setting up arlib environment..."

# 1. Create virtual environment if it doesn't exist
# Shoud we allow for choosing the version of Python (in an os, there may be multiple Python versions)
# Should we allow for using other toolsfor virtual environments, such as conda?
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
(cd "${SCRIPT_DIR}/bin_solvers" && python "download.py")
# cvc5, mathsat, z3.

# 4. Run tests
echo "Running tests..."
if [ -f "${SCRIPT_DIR}/unit_test.sh" ]; then
    bash "${SCRIPT_DIR}/unit_test.sh"
else
    echo "Warning: unit_test.sh not found"
fi

echo "Setup completed successfully!"