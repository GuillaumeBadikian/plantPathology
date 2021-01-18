#!/usr/bin/env bash

[[ $1 == "" ]] && venv_name="venv" || venv_name=$1


echo "activate venv"

sourceF="./src/projet.py"
venv="${venv_name}/bin/activate"

source $venv

python $sourceF

deactivate



