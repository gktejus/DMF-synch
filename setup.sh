#!/bin/bash
pip install -r requirements.txt
gdown --id 1RTiEFxRK4ub4D-VlUkS9jX_N5AGT6VJF
unzip dataset.zip -d datasets/
rm data.zip 
mkdir logs