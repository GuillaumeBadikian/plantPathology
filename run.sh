#!/usr/bin/env bash
echo "activate venv"

sourceF="./src/projet.py"
venv="./mon_env_virtuel/bin/activate"

source $venv

python $sourceF


