pip install -r requirements.txt
mkdir datasets
gdown --id 17rVoMcJOS-8Zh_0R82Qm4Z7aIpenikBE
gdown --id 1-0ZwF8OnoGrtsp7ZGoVgl3gk0J9Gx5hb
gdown --id 1_FDM2tXNssgTFPHtTuOy1yx63szGYM45
gdown --id 1S0ww2h0mtByLiwhk9MrEgsCvmUL6BdEO
unzip mat_synth.zip -d datasets/
unzip dataset_synthetic -d MATLAB_SO3/
unzip data.zip -d datasets/
unzip matlab_data.zip -d MATLAB_SO3/
rm -rf data.zip 
rm -rf matlab_data.zip
rm -rf dataset_synthetic.zip
rm -rf mat_synth
mkdir logs