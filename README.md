# CS6476 Project
https://aparna-1407.github.io/cs6476_project_team18/

## Setup
1. Create a conda environment and activate it
2. pip install -r requirements.txt
3. mkdir models
4. Download IP Adapter models and copy it to the models directory
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
    download image_encoder folder from https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder
   
## File Organisation
- Code Files: You can find code files in `./src/`
- Example outputs: You can find the example outputs organized as sources, poses and generated outputs in `./data/examples`


## Command to run
```
cd src
python pose_transfer.py
```
## Text guided image editting:
```
cd stable-diffusion
. .venv/bin/activate
```
Run the jupyter notebook under `stable-diffusion/notebooks/imagic.ipynb`
To instantiate the jupyter notebook within the created venv:
```
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
python -m ipykernel install --user --name venv --display-name "Python (sd)"
```
Ensure the jupyter notebook is using the Python(sd) kernel

Inside the stable-diffusion submodule, make sure to clone the module using
```
cd text\ guided\ image\ editing
git clone -e https://github.com/justinpinkney/stable-diffusion.git
```
Download Stable diffusion check point to run the code
```
cd text\ guided\ image\ editing/stable-diffusion/models/ldm
mkdir stable-diffusion-v1
cd stable-diffusion-v1
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
```
Copy our notebook to stable-diffusion submodule
```
cp notebooks/imagic_custom.ipynb text\ guided\ image\ editing/stable-diffusion/notebooks/
```
