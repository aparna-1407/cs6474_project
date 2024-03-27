---
layout: default
---
# PoseMaster Fusion: Transforming Images by capturing Pose Dynamics
## Contents
1. [Introduction](#introduction)
2. [Previous Work](#previous-work)
3. [Approach](#approach)
4. [Experiments and Results](#experiments-and-results)
5. [Future Steps](#future-steps)
6. [Contribution](#contribution)
7. [References](#references)

## Introduction
<p align="justify">
PoseMaster Fusion is a diffusion based model for 2D human pose retargeting. Specifically, given a reference image, the model aims to generate a set of new images of the person by controlling the poses and other features such as the style while keeping the identity unchanged. At its core, this project seeks to revolutionize the way we interact with and manipulate images, providing users with unprecedented control over the pose dynamics of their digital images. By leveraging the ability of control nets to understand and map poses from reference images or doodles, and harmonizing this with the generative capabilities of stable diffusion models, Pose Master Fusion attempts to synthesize images disentangling appearance from pose and style. 
</p>
<p align="center">
  <img width="405" alt="image" src="https://github.com/aparna-1407/cs6474_project/assets/93538009/f22435b9-2d74-45f6-a653-9717402bb401">
</p>
  
This image can better explain what this model aims to accomplish- given a reference pose that can either be extracted from a doodle or a reference image, the model aims to translate that pose into the source image provided. PoseMaster Fusion allows you to control style in addition to just poses using text prompts to influence the style of your generated image.

<p align="center">
<img width="378" alt="image" src="https://github.com/aparna-1407/cs6474_project/assets/93538009/32fcffa9-6577-4d79-a40e-5f878a7b7bfe">
</p>

### Why is there a need for such a model?

<p>
This exploration is not just a technical endeavor but a response to the growing demand for more intuitive, powerful tools in digital art and media production. As content creation increasingly grows, there's a palpable need for technologies that can simplify complex processes like pose manipulation and style trasnfer, making them quick, simple and accessible. The ability to seamlessly transfer poses from one image to another, with minimal effort and high fidelity such as from doodles and text prompts, opens up new avenues for creativity and efficiency in digital content creation. While translating poses from reference images using diffusion models is a growing area pf research, we through PoseMaster Fusion attempt to extend pose transfer from using just reference images to even include doodles from which poses can be extracted and also envision to include a module that allows enhancements to style via text prompts.
</p>
<p>
The challenges in creating a cohesive system that can accurately interpret human poses, translate them into a different context while retaining the original image's essence, and produce results that are both visually stunning and contextually appropriate are immense. Through PoseMaster Fusion, we attempt to create a model that generalizes well and enables robust appearance control while leveraging the prior knowledge of Stable Diffusion models and ControlNet.
</p>

## Previous Work
### Controlled Image Synthesis
<p style="font-weight:bold">[1]"MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing" by Cao et al.(2023)</p><p> propose to change pose, view, structures, and non-rigid variances of the source image while maintaining the characteristics, texture, and identity by converting existing self attention in diffusion models into mutual self-attention, so that it can query correlated local contents and textures from source images for consistency. To further alleviate the query confusion between foreground and background, they propose a mask-guided mutual self-attention strategy, where the mask can be easily extracted from the cross-attention maps in diffusion models to separate foreground and background. Thye change the non-rigid attributes (e.g., changing object pose) using only text prompts. The pipeline proposed is that the user feeds an image prompt Ps to describe the source image, a modified target prompt Pt (changing just one word in the source prompt) and provides the source image. The mutual attention mechanism queries image content so that it can generate consistent images under the modified target prompt. The mutual attention and cross attention layer extracts informations about features relevant for the target prompt from its knowledge and synthesize a semantic layout. The denoising U-Net injects this information to the source retaining consistency because of the separation of foreground and background. This pipeline that is solely controlled by text prompts along with Stable Diffusion was observed to be ineffective, so T2I adapters were also integrated for more stable synthesis. The major disadvantages are being limited by the knowledge of stable diffusion to generate the target prompt and impact of artifacts or change in background or other inconsistencies with the target prompt. 
</p>

#### Takeaway:
<p>
We believe we can improve our model by incorporating the idea of poses instead of attention maps, avoiding text guidance and focusing on just pose driven image synthesis, integrating ControlNet and T2I adapters as they are superior in image editing than just prompt guide image synthesis by Stable Diffusion and finetuning the model on our dataset to adapt to artifacts and changes in appearance and improving knowledge.
</p>

<p style="font-weight:bold">[2] "MagicPose: Realistic Human Poses and Facial Expressions Retargeting with Identity-aware Diffusion" by Chang et al.(2024)</p><p> proposes a diffusion model based 2d human pose and facial expression retargetting. To retain identity or consistency of source image they explored connecting the attention layers of diffusion U-Net to provide layer-by-layer attention guidance and retain the apperance of the source image as these layers are highly relevant to the appearance of the generated images.They thus pretain the Stable Diffusion U-Net along with a "Appearance Control Module" that has self attention layers, the key-value pairs are connected together and attention is calculated. ControlNet copies the encoder and middle blocks of SD-UNet, whose output feature maps are added to the decoder of SD-UNet to realize pose control while retaining the appearance. To enhance the entanglement, they fine-tune the Pose ControlNet jointly with our Appearance Control Module this provides enhanced results.
</p>

#### Takeaway:
<p>
We believe we can improve our model by incorporating the appearance control module and finetuning our ControlNet with it.
</p>

### Multimodal Image Generation
<p style="font-weight:bold">
  [3]“UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion” by Wei Li et al.(2024)</p><p> presents zero-shot multi-entity subject-driven generations through multimodal instructions or MLLM framework-a conditional denoising diffusion network for generating images based on the encoded multimodal input. They leverage a two-step training process: First is text-to-image training using a denoising diffusion UNet architecture and conditioning it on the text using a cross-attention mechanism. Then, the multimodal instruction tuning is achieved by training using millions of pairs of multimodal prompts created using DINO and SAM based data processing pipeline to improve the capability of multimodal generation.
</p>

#### Takeaway:
<p>
  The efficacy of the text-to-image pretraining on the denoising U-Net shown by this work inspires us to try this out for our text based image enhancement and style edit endeavour.
</p>
<p style="font-weight:bold">
  [4]“BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing” by SalesForce AI Research(2023)</p><p> introduces a new subject-driven image generation model that supports multimodal control, taking images and text prompts as subject inputs. It trains a multimodal encoder to extract visual features from the subject image and the multimodal module to provide subject representations that align with the text. It also uses Stable DIffusion for learning this subject representation and producing it, CLIP is the text encoder that generates the text embeddings, and 2 modules of BLIP-2 a frozen pre-trained image encoder to extract generic image features, and a multimodal encoder (i.e. Q-Former) for image-text alignment. The pretrained U-Net is connected with ControlNet for structure control and it combines subject prompt embeddings with text prompt embeddings for multimodal controlled generation where cross attention maps create a mask for regions to be edited. This is an efficient zero-shot subject-driven image generation. 
</p>

#### Takeaway:
<p>
  We will also attempt to use cross attention control, i.e, subject embeddings along with text embeddings to guide which portions of the image to edit based on the text prompt. 
</p>

### Image Generation Models
<p style="font-weight:bold">
  [5]“Denoising Diffusion Probabilistic Models”by Ho et.al(2020)</p><p> introduces a new class of generative models called Denoising Diffusion Probabilistic Models (DDPMs) for image synthesis. DDPMs generate high-quality images by gradually adding noise to an initial random noise image and then "denoising" it back to a realistic image through a series of steps.
</p>

#### Takeaway:
<p>
  We understand that DDPM-based frameworks suit text guided image generation tasks best. 
</p>

<p style="font-weight:bold">
  [6]“InstructPix2Pix Learning to Follow Image Editing Instructions” by Brooks et al(2023)</p><p> suggest DDPMs are powerful models for generating high-quality images, but they usually struggle with tasks like image editing due to their inherent noise removal process. This paper proposes InstructPix2Pix, a DDPM-based framework that leverages text instructions to guide the editing process while preserving image details.
</p>

#### Takeaway:
<p>
  We also plan to use InstructPix2Pix module of StableDiffusion for the text guided image editing task. 
</p>

## Approach

### Overall Pipeline
<p align="center">
 <img width="678" alt="image" src="https://github.com/aparna-1407/cs6474_project/assets/93538009/34d0a200-aec0-4246-bbf9-35d6417490f4">
</p>
The overall approach is divided in 3 modules.

1. <p style="font-weight:bold">The Appearance Control Module (ACM)</p><p> which consists of the Stable Diffusion V1.5 and a similar Denoising U-Net where the attention layer outputs are calculated by concatenating the key-value pair values between the corresponding self attention layers. This is done because the output of the self-attention layers is responsible for the appearance of the generated image and adding the auxiliary U-Net and concatinating its key-value pairs is equivalent to reinforcing the source image's appearance.</p>
2.  <p style="font-weight:bold">Appearance-Disentangled Pose Control Module (ADPCM) </p><p> involves finetuning the Stable Diffusion ControlNet pipeline to use the ACM instead of just stable diffusion. This finetuning helps to disentangle pose re-targeting from the appearance. We basically have to finetune these modules with pairs of source image-pose and target image. Using a dataset where images with multiple poses of the same person are available will be used.</p>
3. <p style="font-weight:bold">Text Guided Enhancement Module</p><p>which consists of a text encoder which creates prompt embeddings and we combine this prompt embeddings used with subject/source embeddings (understanding the structure of the source) generated by the InstructPix2Pix model within stable diffusion. The cross attention layers of stable diffusion's cross attention layers uses the embeddings and masks regions to be edited and the prompt embeddings guides the editing. This will leverage the prior knowledge of stable diffusion and not require any finetuning.</p>

> For Project Update-1 we have completed part of the Pose Control Module. We have identified how to set up the pipeline corrently and how ControlNet works best with Stable Diffusion to generate the right output. More details on the different approaches attempted will be explained in next sections. Only during the set up of our pipeline did we realize that ControlNet doesn't preserve appearance and based on further research arrived at the current architecture. So our future steps would be to build a fine-tuned ACM and then incorporate within the pipeline to create the ADPCM.

## Experiments and Results
(Work done for Project Update-1)
Following is a summary of all the approaches we tried, what we observed, and the challenges faced.
### Dataset
**[Deep Fashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)** 
<p> 
  We use the in-shop retrieval subset of the Deep Fashion dataset which consists of 52,712 high-resolution images of fashion models with diverse backgrounds,viewpoints,and lighting conditions. It contains the images of the same model available in different poses thus enabling the finetuning and training stages we plan to execute. We extract the skeletons using OpenPose. The model ID of the training and testing sets donot overlap. 
</p>

### Experiments
##### Using the prior knowledge of ControlNet and Stable Diffusion

We use Stable Diffusion(SD) V1.5 and ControlNet (compatible with SDv1.5) directly with no finetuning due to the extensive knowledge and generalization capabilities inherent in these models. These pre-trained models have been exposed to diverse datasets, enabling them to handle a wide variety of scenarios without the need for additional training. By leveraging the learned representations we attempt to save significant resources and time, while trying to achieve accurate results.

However, ControlNet takes a pose image and a text prompt and generates a new image aligned with the text prompt that matches the pose represented in the pose image. This is not ideal for our use case as we want to achieve pose transfer for a source image given a reference pose image. Additionally for future experiments, we also want to leave functionality for aligning generations with text prompts as well in addition to the source image. To achieve this, we need to inject functionality for image prompt input for the source image, along with the default input functionality for the text prompts and the pose image in the ControlNet module. To explore how this can be achieved, we tried the following approaches:



**Background: `StableDiffusionControlNetPipeline`**
<div style="color: gray">
This pipeline uses the OpenPose ControlNet module along with SDv1.5 in half precision (float16) and the UniPCMultistepScheduler which is the fastest diffusion model scheduler. The pipeline essentially uses CLIP text encoder for processing text prompts which creates text embeddings, then the SD UNet which processes the text prompt and the source image and generates subject embeddings from the image followed by ControlNet which transforms the source image using the reference image, then the VAE decoder which finally converts the latent information into a generated image. 

* This pipeline works well when input is simply a textprompt about the image we want and the reference pose image because SD generate images using a prompt and the latent representation coming from its prior knowledge is easy to perceive and manipulate by ControlNet.

<p align="center">
<img width="1000" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/42696e7c-8001-4759-ad32-d7b91c1d093c">
</p>

</div>

**Method 1: Stable Diffusion ControlNet Reference Pipeline**

* We create a custom pipeline that hacks the default `StableDiffusionControlNetPipeline` from the `diffusers` library to allow image generation using an additional reference image (source image) along with the pose and text prompt inputs. 

* This is achieved by hacking into the self-attention and group normalization computations within the U-Net architecture, introducing custom forward methods that integrate the reference image information. Specifically, the self-attention mechanism is modified to consider the reference image as an additional encoder state, while the group normalization layers are adjusted to utilize the statistics of the reference image. These modifications allow for better control over the generated image's content and style, guided by the reference image.

* This pipeline doesn't generate good results when we supply a source image to SD along with a text guidance and a reference image  to ControlNet because SD's latent representation are not from it is prior knowledge and the cross attention maps produced are relatively harder to identify and manipulate the control points.
<p align="center">
<img width="1413" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/f28b37f4-ce31-4346-9c1a-72b63e68ec4f">
<img width="1413" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/b08ad061-79e7-4926-bf73-4cb5e7c9b9d8">
</p>

**Method 2: IP-Adapter for ControlNet**

* [IP-Adapter](https://arxiv.org/pdf/2308.06721.pdf) is a lightweight and efficient method to add image prompt capability to pre-trained text-to-image diffusion models. It employs a decoupled cross-attention mechanism that separately processes text and image features, allowing for multimodal image generation. With only 22M parameters, IP-Adapter achieves comparable results to fully fine-tuned models and can be easily integrated with existing structural control tools. This makes it an ideal candidate for getting a baseline model for our use case.

* The implementation for this method uses an IP Adapter wrapper over our `StableDiffusionControlNetPipeline` to add functionality for the image prompts. We use the pre-trained [Controlnet - v1.1 - openpose Version](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose) for developing this baseline.

* The qualitative results from our preliminary experiments (shown below) show that the IP-Adapter is able to generate images that are aligned with the given pose and the source image. However, the generated images sometimes have undesirable variations in fine details present in the source image. For example, the clothes or shoes of the subject in the generated image may not match the source image. This is likely due to the fact that the IP-Adapter is not fine-tuned on our dataset and hence does not have the necessary knowledge to preserve the appearance of the source image. While for most use-cases of pose-transfer, preserving such level of detail is not necessary, it is still important for a good pose transfer model to retain the appearance of the source image as much as possible.
  
<p align="center"> 
<img width="1387" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/592c672e-4192-4b24-836f-9d16272371b1">
</p>



We will aim to address this issue with a improved pose-transfer model that we will develop in the next steps of our project.

### Metrics
We plan to conduct a comprehensive evaluation of the finally developed model using the following metrics.
 * [Structural Similarity Index Measure(SSIM)](https://arxiv.org/pdf/2004.01864.pdf)
 * [Learned Perceptual Image Patch Similarity(LPIPS)](https://arxiv.org/pdf/2401.02414.pdf)
 * [Fréchet Inception Distance (FID)](https://huggingface.co/docs/diffusers/en/conceptual/evaluation)
 * [Clip score](https://huggingface.co/docs/diffusers/en/conceptual/evaluation)
   
## Future Steps

| Tasks | Anticipated Date of Completion |
|:------|:-------------------------------|
| Design an Appearance Control Module using a pretrained SD U-Net and auxiliary U-Net. | Mar 31 |
| Finetune ControlNet with the Appearance Control Module, freezing the Auxiliary U-Net to preserve appearance, thus creating an Appearance Disentangled Pose Control Module. | Apr 4 |
| Understand and incorporate text-guided enhancements via the InstructPix2Pix module of SD. | Apr 8 |
| Evaluate the entire model's results based on the decided metrics. | Apr 12 |
| Examine the model's generalizability on other datasets, such as the frames from the [TikTok dataset](https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset), if possible. | Apr 16 |

## Contribution

| Tasks | Member |
|:------|:-------|
| Design an Appearance Control Module using a pretrained SD U-Net and auxiliary U-Net. | Aparna and Shubham |
| Finetune ControlNet with the Appearance Control Module, freezing the Auxiliary U-Net to preserve appearance, thus creating an Appearance Disentangled Pose Control Module. | Poojitha |
| Understand and incorporate text-guided enhancements via the InstructPix2Pix module of SD. | Poojitha |
| Evaluate the entire model's results based on the decided metrics. | Shubham |
| Examine the model's generalizability on other datasets, such as the frames from the [TikTok dataset](https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset), if possible. | Aparna |
| Update the website and maintain the repository. | All |






