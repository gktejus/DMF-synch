pip install -r requirements.txt
mkdir datasets
gdown --id 17rVoMcJOS-8Zh_0R82Qm4Z7aIpenikBE
gdown --id 1-0ZwF8OnoGrtsp7ZGoVgl3gk0J9Gx5hb
unzip data.zip -d datasets/
unzip matlab_data.zip -d MATLAB_SO3/
rm -rf data.zip 
rm -rf matlab_data.zip
mkdir logs