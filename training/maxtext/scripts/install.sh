#!/bin/bash

sudo apt install python3.12-venv -y

python -m venv 
conda activate maxtext

pip install -r requirements.txt