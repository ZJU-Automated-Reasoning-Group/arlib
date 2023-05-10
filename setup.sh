
# install venv

# create a venv

# install packages
pip3 install -r requirements.txt

# download some binary solvers

# patch z3 (e.g., In goal.dimacs(), do not print some comments)
wget https://github.com/Z3Prover/z3/archive/refs/tags/z3-4.12.0.zip
unzip z3-4.12.0.zip
rm z3-4.12.0.zip
cd cd z3-z3-4.12.0
mkdir build
cd build
cmake .. -DZ3_BUILD_PYTHON_BINDINGS=true -DZ3_INSTALL_PYTHON_BINDINGS=true -DCMAKE_INSTALL_PYTHON_PKG_DIR=venv
