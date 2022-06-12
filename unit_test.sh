# export PYTHONDONTWRITEBYTECODE=True

# use -rP show display the output of the tests (print(xxx))
# Skip slow tests (-m "not slow")
# Exit on error (-x)
# Rule of thumb: if a test takes more than 10 seconds it
#                should be marked as slow using:
#                    @pytest.mark.slow
python3 -m pytest -m "not slow" -x pdsmt/tests