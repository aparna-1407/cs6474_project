from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from types import MethodType
from ip_adapter import IPAdapter

from stable_diffusion_controlnet_reference import StableDiffusionControlNetReferencePipeline

CONTROL_NET_MODEL = "fusing/stable-diffusion-v1-5-controlnet-openpose"
CONTROL_NET_MODEL_IP_ADAPTER = "lllyasviel/control_v11p_sd15_openpose"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
VAE_MODEL_PATH = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "../models/image_encoder/"
ip_ckpt = "../models/ip-adapter_sd15.bin"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

class PoseTransfer():
    def __init__(self, controlnet_model=CONTROL_NET_MODEL, stable_diffusion_model=MODEL_ID):
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16
        )
        self.model_id = stable_diffusion_model

            
    def get_pose(self, imgs):
        model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        poses = [model(img) for img in imgs]
        return poses

    def run_pose_transfer_ip_adapter(self,input_images, ref_images):
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            )
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(dtype=torch.float16)
        poses = self.get_pose(input_images)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )
        ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device="cuda")
        for i in range(len(poses)):
            # Change num_samples to 1 to output 1 image for 1 pose
            images = ip_model.generate(pil_image=ref_images[0], image=poses[i], 
            width=512, height=768, num_samples=4, num_inference_steps=50, seed=42)
            # grid = image_grid(images, 1, 4)
            for _ix, img in enumerate(images):
                img.save(f'images/ip_adapter_image_{i+_ix + 1}.png')
        return


    def run_pose_transfer(self, input_images, ref_images):
        pipe =  StableDiffusionControlNetReferencePipeline.from_pretrained(
            self.model_id,
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        prompt = "1 person, pose, high quality, extremely detailed"
        negative_prompt = "lowres, bad anatomy, worst quality, low quality"
        poses = self.get_pose(input_images)

        k = len(poses)
        generator = [torch.Generator(device="cpu").manual_seed(5) for i in range(1)]

        for i in range(len(poses)):
            output = pipe(
                prompt = prompt,
                image = poses[i],
                ref_image= ref_images[0],
                negative_prompt = negative_prompt,
                generator=generator,
                num_inference_steps=20,
                reference_attn=True,
                reference_adain=True,
                guess_mode=True,
                guidance_scale = 1.1,
                # # num_images_per_prompt = 5,
                attention_auto_machine_weight = 1.0,
                controlnet_conditioning_scale = 0.9
            )
            for _ix, img in enumerate(output.images):
                img.save(f'images/image_{i + _ix + 1}.png')
        # image_grid(output.images, 1, 1)

if __name__=="__main__":
    ref_images = [load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png")]
    # ref_images = [load_image("https://as1.ftcdn.net/v2/jpg/03/11/63/54/1000_F_311635498_i6ouJY7aYwMXd5Mp4qvrZcK6aaMd1v4Z.jpg")]
    imgs = [load_image("https://i.pinimg.com/736x/b4/a1/d4/b4a1d47aca99300039e07e77923c88e2.jpg")]
    # urls = "yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"
    # imgs = [
    #     load_image("https://hf.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url)
    #     for url in urls
    # ]
    pose_transfer = PoseTransfer()
    # pose_transfer.run_pose_transfer(imgs, ref_images)
    pose_transfer.run_pose_transfer_ip_adapter(imgs, ref_images)
