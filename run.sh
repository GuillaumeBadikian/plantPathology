#!/usr/bin/env bash
echo "activate venv"

sourceF="./src/projet.py"
venv="./venv/bin/activate"

source $venv

python $sourceF


