#!/bin/bash
# Birth Simulation Runner

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the simulation
python -m birth.main "$@"
