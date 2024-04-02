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
  


