#!/bin/bash

# Set the required paths
MODEL_CONFIG="model_lib/ControlNet/models/cldm_v15_reference_only_pose.yaml"
IMAGE_PRETRAIN_DIR="./pretrained_weights/model_state-110000.th"
INPUT_IMAGE_PATH="./example_data/image/out-of-domain/001.png"
POSE_MAP_PATH="./example_data/pose_sequence/001/0001.png"
OUTPUT_IMAGE_PATH="output_image.jpg"

# Set optional arguments (if needed)
ETA=0.0
DEVICE="cuda"

# Run the Python script
python image_pose_transfer.py \
    --model_config $MODEL_CONFIG \
    --image_pretrain_dir $IMAGE_PRETRAIN_DIR \
    --input_image_path $INPUT_IMAGE_PATH \
    --pose_map_path $POSE_MAP_PATH \
    --output_image_path $OUTPUT_IMAGE_PATH \
    --eta $ETA \
    --device $DEVICE \
    --control_type body+hand+face \
    --wonoise \
    --extract-pose \
    
