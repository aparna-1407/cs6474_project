import argparse

# torch
import torch
from torchvision.utils import save_image
from torchvision import transforms as T

# utils
from utils.checkpoint import load_from_pretrain
# model
from model_lib.ControlNet.cldm.model import create_model

import torchvision.transforms as transforms
from PIL import Image
# openpose
from model_lib.ControlNet.annotator.openpose import OpenposeDetector
import numpy as np
from PIL import Image

openpose = OpenposeDetector()

def center_crop_to_512(image_path):
    # Read the image from the local path
    image = Image.open(image_path)
    # Define the transformation to center crop to 512x512
    transform = transforms.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(
                    512,
                    scale=(1.0, 1.0), ratio=(1., 1.),
                    interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply the transformation
    cropped_image = transform(image)

    return cropped_image

def center_crop_pose_to_512(image_path, extract_pose_from_image=False):
    # Read the image from the local path
    image = Image.open(image_path)
    image = np.array(image.convert("RGB") if image.mode != "RGB" else image)

    if extract_pose_from_image:
        image = Image.fromarray(openpose(image, hand_and_face=True))

    # Define the transformation to center crop to 512x512
    transform =  T.Compose([
        T.RandomResizedCrop(
                512,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    # Apply the transformation
    cropped_image = transform(image)

    return cropped_image

def load_state_dict(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict  

def get_cond_control(args, input_image, pose_map, device, model=None):
    if args.control_type == "body+hand+face":
        pose_map = pose_map.to(device)
        cond_img_cat = model.get_first_stage_encoding(model.encode_first_stage(input_image))
        return [pose_map], [cond_img_cat]
    else:
        raise NotImplementedError(f"cond_type={args.control_type} not supported!")

def main(args):
    # Load the pre-trained model
    model = create_model(args.model_config).cpu()
    model.to(args.device)
    load_state_dict(model, args.image_pretrain_dir, strict=True)

    # Load the input image and pose map
    input_image = center_crop_to_512(args.input_image_path).unsqueeze(0).to(args.device)
    pose_map = center_crop_pose_to_512(args.pose_map_path,
                                       extract_pose_from_image=args.extract_pose
                                    ).unsqueeze(0).to(args.device)

    # Get the control maps
    c_cat_list, cond_img_cat = get_cond_control(args, input_image, pose_map, args.device, model=model,)

    model.eval()

    text = [""] if args.text_prompt is None else [args.text_prompt]
    c_cross = model.get_learned_conditioning(text)
    uc_cross = model.get_unconditional_conditioning(1)

    pose_input = c_cat_list[0]

    c = {"c_concat": [pose_input], "c_crossattn": [c_cross], "image_control": cond_img_cat, "wonoise": args.wonoise, "overlap_sampling": False}
    uc = {"c_concat": [pose_input], "c_crossattn": [uc_cross], "wonoise": args.wonoise, "overlap_sampling": False}

    noise_shape = (1, model.channels, model.image_size, model.image_size)
    noise = torch.randn(noise_shape).cuda()

    with torch.no_grad():
        output_image, _ = model.sample_log(
            cond=c,
            batch_size=1,
            ddim=True,
            ddim_steps=50, eta=args.eta,
            unconditional_guidance_scale=7,
            unconditional_conditioning=uc,
            inpaint=None,
            x_T=noise,                                        
        )
        output_image = model.decode_first_stage(output_image)

    # Save the output image
    save_image(output_image.clamp(-1, 1).cpu().add(1).mul(0.5), args.output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file')
    parser.add_argument('--image_pretrain_dir', type=str, required=True, help='Path to the pre-trained image model checkpoint')
    parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--pose_map_path', type=str, required=True, help='Path to the pose map')
    parser.add_argument('--output_image_path', type=str, required=True, help='Path to save the output image')
    parser.add_argument('--eta', type=float, default=0.0, help='Value of eta for DDIM sampling')
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='Feed text_prompt into the model')
    parser.add_argument('--wonoise', action='store_true', default=False,
                        help='Use with referenceonly, remove adding noise on reference image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--control_type', type=str, nargs="+", default=["body+hand+face"],
                        help='The type of conditioning')
    parser.add_argument('--extract-pose', action='store_true', help='Extract pose from the input image')

    args = parser.parse_args()
    main(args)