---
layout: default
---
# PoseMaster Fusion: Transforming Images by capturing Pose Dynamics
## Contents
1. [Introduction](#introduction)
  * [Project Description](#project-description)
  * [Motivation](#why-is-there-a-need-for-such-a-model-?)
2. [Related Work](#related-work)
  * [Explaining Context](#explaining-context)
  * [Our Project in Context](#our-project-in-context)
3. [Approach](#approach)
  * [Overall Pipeline- Method Overview](#overall-pipeline)
  * [Contribution](#contribution)
  * [Rationale for Success- Intuition](#rationale-for-success)
4. [Experiments Setup](#experiments-setup)
  * [Experiments](#experiments)
  * [Input Description](#input-data)
  * [Desired Output](#output-expected)
  * [Metrics](#metrics)
5. [Results](#results)
  * [Key Results](#final-pipeline-results)
  * [Baseline Comparison](#comparison-with-baseline)
  * [Performance](#performance)
6. [Discussion](#discussion)
7. [Challenges](#challenges)
8. [Team Member Contribution](#team-member-contribution)
8. [References](#references)

## Introduction
### Project Description
<p align="justify">
PoseMaster Fusion is a diffusion based model for 2D human pose re-targeting. Specifically, given a reference image, the model aims to generate a set of new images of the person by controlling the poses and other features such as the style while keeping the identity unchanged. At its core, this project seeks to revolutionize the way we interact with and manipulate images, providing users with unprecedented control over the pose dynamics of their digital images. By leveraging the ability of control nets to understand and map poses from reference images or doodles, and harmonizing this with the generative capabilities of stable diffusion models, Pose Master Fusion attempts to synthesize images disentangling appearance from pose and style. 
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
This exploration is not just a technical endeavor but a response to the growing demand for more intuitive, powerful tools in digital art and media production. As content creation increasingly grows, there's a palpable need for technologies that can simplify complex processes like pose manipulation and style transfer, making them quick, simple and accessible. The ability to seamlessly transfer poses from one image to another, with minimal effort and high fidelity such as from doodles and text prompts, opens up new avenues for creativity and efficiency in digital content creation. While translating poses from reference images using diffusion models is a growing area of research, through PoseMaster Fusion, we attempt to extend pose transfer from using just reference images to even include doodles from which poses can be extracted and also envision to include a module that allows enhancements to style via text prompts.
</p>
<p>
The challenges in creating a cohesive system that can accurately interpret human poses, translate them into a different context while retaining the original image's essence, and produce results that are both visually stunning and contextually appropriate are immense. Through PoseMaster Fusion, we attempt to create a model that generalizes well and enables robust appearance control while leveraging the prior knowledge of Stable Diffusion models and ControlNet.
</p>

## Related Work
### Explaining Context
#### Controlled Image Synthesis
<p style="font-weight:bold">[1]"MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing" by Cao et al. (2023)</p><p> proposes to change pose, view, structures, and non-rigid variances of the source image while maintaining the characteristics, texture, and identity by converting existing self attention in diffusion models into mutual self-attention, so that it can query correlated local contents and textures from source images for consistency. To further alleviate the query confusion between foreground and background, they propose a mask-guided mutual self-attention strategy, where the mask can be easily extracted from the cross-attention maps in diffusion models to separate foreground and background. Thye change the non-rigid attributes (e.g., changing object pose) using only text prompts. The pipeline proposed is that the user feeds an image prompt Ps to describe the source image, a modified target prompt Pt (changing just one word in the source prompt) and provides the source image. The mutual attention mechanism queries image content so that it can generate consistent images under the modified target prompt. The mutual attention and cross attention layer extracts informations about features relevant for the target prompt from its knowledge and synthesize a semantic layout. The denoising U-Net injects this information to the source retaining consistency because of the separation of foreground and background. This pipeline that is solely controlled by text prompts along with Stable Diffusion was observed to be ineffective, so T2I adapters were also integrated for more stable synthesis. The major disadvantages are being limited by the knowledge of stable diffusion to generate the target prompt and impact of artifacts or change in background or other inconsistencies with the target prompt. 
</p>

**`Takeaway:`**
<p>
by focusing more on pose adjustments rather than solely on attention maps and eliminating the reliance on text prompts for guiding image synthesis. By integrating mechanisms similar to ControlNet and T2I adapters, which have shown superior performance in image editing beyond what can be achieved with mere prompt-based synthesis by Stable Diffusion, our model could enhance its editing capabilities. Furthermore, fine-tuning our model on a tailored dataset will allow it to better handle potential artifacts and variations in appearance, thereby expanding its understanding and ability to generate target prompts more effectively.
</p>

<p style="font-weight:bold">[2] "MagicPose: Realistic Human Poses and Facial Expressions Retargeting with Identity-aware Diffusion" by Chang et al. (2024)</p><p> proposes a diffusion model based 2d human pose and facial expression re-targeting. To retain identity or consistency of source image they explored connecting the attention layers of diffusion U-Net to provide layer-by-layer attention guidance and retain the apperance of the source image as these layers are highly relevant to the appearance of the generated images. They thus pretain the Stable Diffusion U-Net along with a "Appearance Control Module" that has self attention layers, the key-value pairs are connected together and attention is calculated. ControlNet copies the encoder and middle blocks of SD-UNet, whose output feature maps are added to the decoder of SD-UNet to realize pose control while retaining the appearance. To enhance the entanglement, they fine-tune the Pose ControlNet jointly with the Appearance Control Module, achieving enhanced results.
</p>

**`Takeaway:`**
<p>
We believe we can improve our model by incorporating the appearance control module and finetuning our ControlNet with it.
</p>

#### Multimodal Image Generation
<p style="font-weight:bold">[3]“UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion” by Wei Li et al. (2024)</p><p> presents zero-shot multi-entity subject-driven generations through multimodal instructions or MLLM framework — a conditional denoising diffusion network for generating images based on the encoded multimodal input. They leverage a two-step training process: First is text-to-image training using a denoising diffusion UNet architecture and conditioning it on the text using a cross-attention mechanism. Then, the multimodal instruction tuning is achieved by training using millions of pairs of multimodal prompts created using DINO and SAM based data processing pipeline to improve the capability of multimodal generation.
</p>

**`Takeaway:`**
<p>
  The efficacy of the text-to-image pretraining on the denoising U-Net shown by this work inspires us to try this out for our text based image enhancement and style edit endeavour.
</p>
<p style="font-weight:bold">[4]“BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing” by SalesForce AI Research (2023)</p><p> introduces a new subject-driven image generation model that supports multimodal control, taking images and text prompts as subject inputs. It trains a multimodal encoder to extract visual features from the subject image and the multimodal module to provide subject representations that align with the text. It also uses Stable DIffusion for learning this subject representation and producing it, CLIP is the text encoder that generates the text embeddings, and 2 modules of BLIP-2 a frozen pre-trained image encoder to extract generic image features, and a multimodal encoder (i.e. Q-Former) for image-text alignment. The pretrained U-Net is connected with ControlNet for structure control and it combines subject prompt embeddings with text prompt embeddings for multimodal controlled generation where cross attention maps create a mask for regions to be edited. This is an efficient zero-shot subject-driven image generation. 
</p>

**`Takeaway:`**
<p>
  We will also attempt to use cross attention control, i.e, subject embeddings along with text embeddings to guide which portions of the image to edit based on the text prompt. 
</p>

#### Image Generation Models
<p style="font-weight:bold">
  [5]“Denoising Diffusion Probabilistic Models”by Ho et.al (2020)</p><p> introduces a new class of generative models called Denoising Diffusion Probabilistic Models (DDPMs) for image synthesis. DDPMs generate high-quality images by gradually adding noise to an initial random noise image and then "denoising" it back to a realistic image through a series of steps.
</p>

**`Takeaway:`**
<p>
  We understand that DDPM-based frameworks suit text guided image generation tasks best. 
</p>

<p style="font-weight:bold">
  [6]“InstructPix2Pix Learning to Follow Image Editing Instructions” by Brooks et al (2023)</p><p> suggest DDPMs are powerful models for generating high-quality images, but they usually struggle with tasks like image editing due to their inherent noise removal process. This paper proposes InstructPix2Pix, a DDPM-based framework that leverages text instructions to guide the editing process while preserving image details.
</p>

**`Takeaway:`**
<p>
  We also plan to use InstructPix2Pix module of StableDiffusion for the text guided image editing task. 
</p>

### Our Project in Context

* Previous Work in the field of Subject Guided Image Generation used only text prompts or reference image inputs, they did not utilize both modalities.
  * Existing methods gave text inputs to guide SD models to generate an image matching up to the prompt given, for which they finetuned the CLIP Encoder for the model to better understand the text inputs and what parts of the image should be editted.
  * Methods that provided image as the input used ControlNet along with SD to trasfer the pose. But preserving the appearance and enhancing generalizability of these models is still an area of research.
* Our project attempts to combine the visual and language modalities.</p>
  * The user can provide an image A as input, which is the image to be modified, and an image B which contains the pose that A should be modified to replicate.
  * The user can also provide a text prompt on how to edit the original image A, such as adding accessories, changing attributes of appearance, or style of the image.
  * Subject Guided Image Generation and Editting, achieved by our project, has not be addressed by existing works, and we have managed to address this gap through our project.

## Approach

### Overall Pipeline
<p align="center">
  <img width="1012" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/b1295539-50fb-4d97-a373-a0c9b1846819">
</p>
The overall approach is divided in 3 modules.

### Identity-aware Pose Retargeting
1. <p style="font-weight:bold">The Appearance Control Module (ACM)</p><p> which consists of the Stable Diffusion V1.5 and a similar Denoising U-Net where the attention layer outputs are calculated by concatenating the key-value pair values between the corresponding self attention layers. This is done because the output of the self-attention layers is responsible for the appearance of the generated image and the integration of the auxiliary U-Net and concatinating its key-value pairs is equivalent to reinforcing the source image's appearance.</p>

#### Architecture
* This is handled by the `ControlledUnetModelAttnPose` which inherits from [Open AI's guided diffusion U-Net](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py) class which follows the same architecture as the SD UNet. 
  * `ControlledUnetModelAttnPose` replicates the architecture of the SD U-Net which helps in controlling the generation process of pre-trained diffusion model via multi source attention layers, enabling more flexible information interchange among distant pixels, flexible emphasis on certain regions. And therefore it is more suited for the task of pose retargeting.
  * It also extends the capabilities of the parent U-Net architecture, incorporating advanced features of dynamic cross-attention mechanisms using spatial transformers applied at resolutions 1,2 and 4 with a model channel width of 320 and a context dimension of 768 aiding it to better learn the certain original representations in the image as dictated by the middle block of the U-Net.
  * The middle block of the U-Net consists of one or more convolutional layers that process the deepest, most compressed representation of the input data. This block is crucial for capturing the high-level context of the input image, which is then used by the decoder to generate detailed segmentations or manipulations of the input.
* The Appearance Control starts by masking each of the body, face and pose of the input image and as given as input to `ControlledUnetModelAttnPose`'s `forward()` which applies attention on the whole image and stores the attention head representation, and then spatial transformations and attention on the middle block and finally the decoder of the U-Net produces a complete image recovering the masked portion. This way the module learns the original representations of the input image well. 
* The base latent diffusion model used is Stable Diffusion V1.5

2. **Appearance-Disentangled Pose Control Module (ADPCM)** involves finetuning the Stable Diffusion ControlNet pipeline to use the ACM instead of just stable diffusion. This finetuning helps to disentangle pose re-targeting from the appearance. We basically have to finetune these modules with pairs of source image-pose and target image. Using a dataset where images with multiple poses of the same person are available will be used.
#### Architecture
* This is handled by the `ControlledUnetModelAttnPose` and `ControlNetReference` which inherits from [original ControlNet architecture](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py).
  * `ControlledUnetModelAttnPose` has a `pose_control` flag which gets encoded vectors from the `ControlNetReference` module which it concatenates within the feature maps of the original image. The output encoded vectors from `ControlNetReference` convey exactly how the vectors should be embedded within the feature map due to the presence of `hint_channels` in `ControlNetReference`, allowing the model to integrate additional hint or guiding information directly into the generation process. This enhances capabilities of parent ControlNet architecture to exclusively modulate the pose attributes of the original image, while the Appearance Control Model focuses on appearance control. The spatial transformers added to `ControlNetReference` applies attention-driven spatial manipulations, providing flexibility in handling complex spatial relationships in the data, ensuring the pose is translated correctly. 
  * The Appearance Disentangled Pose Control is achieved by jointly fine-tuning the ACM with `ControlNetReference`. The blocks of `ControlledUnetModelAttnPose` associated with embedding ControlNet vectors in feature maps alone is left unfrozen and finetuned with `ControlNetReference` so that appearance of the generated image is preserved while trying to accurately transfer the pose.

### Text Conditioned Image Editing   

3. **Text Guided Enhancement Module** which consists of a text encoder which creates prompt embeddings and we combine this prompt embeddings used with subject/source embeddings (understanding the structure of the source) generated by the InstructPix2Pix model within stable diffusion. The finetuning diffusion process creates an interpolation of the embedding space to understand the prompt generated to describe the input image and the actual text prompt thereby creating an optimized text embedding to indicate parts of the image that should be changed and how. The cross attention layers of stable diffusion's cross attention layers uses the embeddings and masks regions to be edited and the prompt embeddings guides the editing. This will leverage the prior knowledge of stable diffusion and not require any finetuning.
#### Architecture
<p align="center">
<img width="800" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/50ac9d85-990c-4834-acf7-2b9e88854cbb">
</p>

* This architecture has 3 modules- Embedding Optimization, Finetuning Diffusion Model and Interpolating Embedding space. We use the CLIP Text Encoder present within SD and SD V1.4 for this module as it is most stable.
  *  Given the target prompt A and the image B, the CLIP Encoder tries to determine a text embedding x describing image B and a text embedding y for prompt A with an objective of reducing the distance between x and y. The reason for determining close x and y is to minimize the number of edits to make to the image, or retain substantial amounts of the original image while editing only whatever is absolutely necessary. This is achieved by freezing the parameters of the generative diffusion model, and optimizing the CLIP Encoder using the denoising diffusion objective using a DDIM Sampler. While the denoising diffusion objective focuses on how to train a model to reverse the diffusion process accurately, the DDIM sampler offers a way to perform this reverse process more efficiently during inference. It leverages the trained model to produce samples in fewer steps by following an implicit trajectory in the noise space.
  * We fine-tune the diffusion models while freezing the CLIP Encoder to generate image C similar to text embedding x with an objective to reduce the reconstruction loss between image C and image B. This is to synchronise the diffusion model and the encoder to ensure that the generation process targets the areas of the image as necessary, in order to find a point that achieves both fidelity to the input image and target text alignment.
  * Since the generative diffusion model is now trained to fully recreate the input image B at the optimized embedding x, we use it to apply the desired edit A by advancing in the direction of the target text embedding y. We apply the base generative diffusion process SD 1.4 to generate an low-resolution edited image capturing features of A, which is then super-resolved using the fine-tuned SD backbone, conditioned on the target text y. This generative process outputs our final high resolution edited image D. The finetuned SD backbone retains the details of the original image and retargets regions specified by target embedding y. The new editted features of the image are supplied by the base SD1.4 as it has vast knowledge to generate the desired edit. 


> For Project Update-1, we completed part of the Pose Control Module. We identified how to set up the pipeline corrently and how ControlNet works best with Stable Diffusion to generate the right output. More details on the different approaches attempted are explained in the next sections. We first evaluated vanilla ControlNet for appearance control, we found that ControlNet is not able to maintain the appearance when generating human images of different poses, making it unsuitable for the re-targeting task. Based on further research and experimentation, we arrived at the current architecture.

### Contribution
The MagicPose[2](https://arxiv.org/pdf/2311.12052.pdf) is the current SOTA in Identity-aware Diffusion and retrageting Poses. We have followed the approach presented by this paper for appearance controlled pose transfer. Following MagicPose, we managed to achieve pose and expression retargeting with a superior capacity to generalize over diverse human identities, however, what we found lacking was subject conditioned image editing which could provide a complete image editing experience to the user. So our contribution to build on MagicPose is to integrate a Text-Conditioned Image Editing pipeline that performs non-trivial semantic edits to real photos in a seamless fashion. Inspired by [Imagic](https://arxiv.org/pdf/2210.09276.pdf), we have created a pipeline which when given only an
input image to be edited and a single text prompt describing the target edit, can perform sophisticated non-rigid edits resulting in a high-resolution image output which aligns well with the target text, while preserving the overall structure, and composition of the original image. Thus, we combined approaches presented by two prior methods in addition to experimenting on new data to create a new end to end pipeline which achieves **Text- Conditioned Subject Guided Image Generation** distinguishes our project within the landscape of image editing and generation and our pipeline also shows promising results.

### Rationale for Success
* Leveraging the strengths of both Stable Diffusion V1.5 for its latent space manipulation and ControlNet for precise pose transfer, our method integrates an Appearance Control Module (ACM) and an Appearance-Disentangled Pose Control Module (ADPCM) to finely tune the balance between maintaining the subject's appearance and accurately transferring poses. This dual-module strategy effectively addresses the challenges of preserving identity and characteristic details during pose alterations, a significant advancement over previous methods that couldn't achieve identity-aware diffusion.
  * Central to this methodology is the `ControlledUnetModelAttnPose`, which employs dynamic cross-attention mechanisms and spatial transformers to ensure modifications are contextually coherent. This not only enhances the model's ability to focus on specific image regions dictated by the pose and appearance controls but also facilitates a more flexible interchange of information among distant pixels, elevating the quality and precision of the output images.
* Moreover, the Text Guided Enhancement Module introduces a novel aspect to our project overcoming a gap that previous methods faced, enabling semantic editing that goes beyond structural adjustments to include stylistic and attribute changes as directed by textual prompts. This module's ability to interpolate between the original image description and the desired edits allows for semantically meaningful modifications, aligning the final output closely with user intentions.


## Experiment Setup
### Experiments

#### Using the prior knowledge of ControlNet and Stable Diffusion

We use Stable Diffusion(SD) V1.5 and ControlNet (compatible with SDv1.5) directly with no finetuning due to the extensive knowledge and generalization capabilities inherent in these models. These pre-trained models have been exposed to diverse datasets, enabling them to handle a wide variety of scenarios without the need for additional training. By leveraging the learned representations we attempt to save significant resources and time, while trying to achieve accurate results.

However, ControlNet takes a pose image and a text prompt and generates a new image aligned with the text prompt that matches the pose represented in the pose image. This is not ideal for our use case as we want to achieve pose transfer for a source image given a reference pose image. Additionally for future experiments, we also want to leave functionality for aligning generations with text prompts as well in addition to the source image. To achieve this, we need to inject functionality for image prompt input for the source image, along with the default input functionality for the text prompts and the pose image in the ControlNet module. To explore how this can be achieved, we tried the approaches summarized below.


**Background: `StableDiffusionControlNetPipeline`**

<div style="color: gray">

This pipeline uses the OpenPose ControlNet module along with SDv1.5 in half precision (float16) and the UniPCMultistepScheduler which is the fastest diffusion model scheduler. The pipeline essentially uses CLIP text encoder for processing text prompts which creates text embeddings, then the SD UNet which processes the text prompt and the source image and generates subject embeddings from the image followed by ControlNet which transforms the source image using the reference image, then the VAE decoder which finally converts the latent information into a generated image. 

* This pipeline works well when input is simply a text prompt about the image we want and the reference pose image because SD generate images using a prompt and the latent representation coming from its prior knowledge is easy to perceive and manipulate by ControlNet.

<p align="center">
<img width="1000" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/42696e7c-8001-4759-ad32-d7b91c1d093c">
</p>

</div>

However, we want to pose transfer for a specific given source image. Mentioned below are a couple of methods we tried to achieve this.

**Experiment 1: Stable Diffusion ControlNet Reference Pipeline**

* We create a custom pipeline that hacks the default `StableDiffusionControlNetPipeline` from the `diffusers` library to allow image generation using an additional reference image (source image) along with the pose and text prompt inputs. 

* This is achieved by hacking into the self-attention and group normalization computations within the U-Net architecture, introducing custom forward methods that integrate the reference image information. Specifically, the self-attention mechanism is modified to consider the reference image as an additional encoder state, while the group normalization layers are adjusted to utilize the statistics of the reference image. These modifications allow for better control over the generated image's content and style, guided by the reference image.

* This pipeline doesn't generate good results when we supply a source image to SD along with a text guidance and a reference image  to ControlNet because SD's latent representation are not from it is prior knowledge and the cross attention maps produced are relatively harder to identify and manipulate the control points.
<p align="center">
<img width="1413" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/f28b37f4-ce31-4346-9c1a-72b63e68ec4f">
<img width="1413" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/b08ad061-79e7-4926-bf73-4cb5e7c9b9d8">
</p>

**Experiment 2: IP-Adapter for ControlNet**

* [IP-Adapter](https://arxiv.org/pdf/2308.06721.pdf) is a lightweight and efficient method to add image prompt capability to pre-trained text-to-image diffusion models. It employs a decoupled cross-attention mechanism that separately processes text and image features, allowing for multimodal image generation. With only 22M parameters, IP-Adapter achieves comparable results to fully fine-tuned models and can be easily integrated with existing structural control tools. This makes it an ideal candidate for getting a baseline model for our use case.

* The implementation for this method uses an IP Adapter wrapper over our `StableDiffusionControlNetPipeline` to add functionality for the image prompts. We use the pre-trained [Controlnet - v1.1 - openpose Version](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose) for developing this baseline.

* The qualitative results from our preliminary experiments (shown below) show that the IP-Adapter is able to generate images that are aligned with the given pose and the source image. However, the generated images sometimes have undesirable variations in fine details present in the source image. For example, the clothes or shoes of the subject in the generated image may not match the source image. This is likely due to the fact that the IP-Adapter is not fine-tuned on our dataset and hence does not have the necessary knowledge to preserve the appearance of the source image. While for most use-cases of pose-transfer, preserving such level of detail is not necessary, it is still important for a good pose transfer model to retain the appearance of the source image as much as possible.
  
We will aim to address this issue with a improved pose-transfer model that we will develop in the next steps of our project.

<p align="center"> 
<img width="1280" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/592c672e-4192-4b24-836f-9d16272371b1">
<img width="1280" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/8b5dca3e-ce3e-4914-9604-850838ceb8e1">
</p>

**Experiment 3: Identity-aware Pose Retargeting**
* We use the pre-trained [MagicDance](https://github.com/Boese0601/MagicDance) model which was originally to be used for re-targeting source images to videos given a pose sequence for the video frames. We re-purpose this model to do pose-transfer for a single source image given a reference image for the pose.

* To achieve this, we first use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract the pose-map for the body, face, and hands from the given reference image, and then use a modified pipeline of the MagicDance model to generate the pose-transferred image. 

* [#TODO: add some comments on the experiment observations and results]

**Experiment 4: Text Conditioned Image Editing**
* Loading in a pretrained Stable Diffusion v1.4 model, the diffusion process is performed in the latent space (of size 4ˆ64ˆ64) of its pre-trained autoencoder, working with 512ˆ512-pixel images.
* We optimize the latent space by freezing the SD backbone and finetuning the CLIPEncoder for 1000 steps with a learning rate of 2e^3 using Adam.
* Then, we fine-tune the diffusion model after freezing the CLIPEncoder for 1500 steps with a learning rate of 5e^7 using Adam.
![image](https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/517ca014-e1ac-43f8-b644-e7dfd58d2ebb)
![image](https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/167814f7-5747-4123-9521-ba78e721af95)

### Input Data
* Our method’s input is a single image as the source image, another image to transfer pose from, and a simple text prompt describing the desired edit, and aims to apply this edit while preserving a maximal amount of details from the image.
* The ACM and ADPCM is trained and finetuned respectively on the [Tiktok dataset](https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset). The Text Conditioned Image Editing uses the existing knowledge of Latent Diffusion Models
  * The TikTok dataset aims to provide data on high fidelity human depths by leveraging a collection of 350 single-person social media dance videos, which are 10-15 seconds long, scraped from the TikTok.
  * The videos are then broken down into their RGB images at 30 frame per second to avoid motion blur, resulting in around 160K images.  
* For our experiments and results, we have finetuned and worked using the [Deep Fashion dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) that contain the images of various models in various poses, along with text annotations that help build our method. 
  * There are 44,096 images from DeepFashion dataset, each image is of size 750×1101 and each image is of jpg format. These images occupy 5.4 GB of size. The dataset also contain textual captions for each image in json format occupying 11 MB of disk space.  
  * The dataset contains labels for various images categorized based on the outfit categories (Denim, Dresses etc.) of the fashion models. Every image has manual annotations for various attributes such as clothes, shapes and textures, dense poses, and textual description for each image as well.
  * We use the in-shop retrieval subset of the Deep Fashion dataset which consists of 52,712 high-resolution images of fashion models with diverse backgrounds,viewpoints,and lighting conditions. It contains the images of the same model available in different poses thus enabling the finetuning and training stages we plan to execute. We extract the skeletons using OpenPose. The model ID of the training and testing sets donot overlap.
* We have also used Self-collected Out-of-Domain Images from online resources. We use them to test our method’s generalization ability to in-the-wild appearance.

### Expected Output

<img width="912" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/13400691/8c49e01c-8e5c-48c4-b130-12deb84f7dc8">
Here is the expected output of our model.
The output of our model would be an image that reflects the text prompt and also adheres to the pose of the reference image. 

### Metrics
The metric of success we can use on our finally developed model is:
 * [Structural Similarity Index Measure(SSIM)](https://arxiv.org/pdf/2004.01864.pdf)
 * SSIM takes a value between 0 to 1, where 0 implies no similarity and 1 implies exact match.
   * The Structural Similarity Index Measure (SSIM) is an advanced metric used to assess the quality of images and videos, particularly in comparison to a reference image. Unlike simpler metrics like Mean Squared Error (MSE) or Peak Signal-to-Noise Ratio (PSNR) that compute absolute errors, SSIM considers changes in structural information, texture, luminance, and contrast, which are more aligned with human visual perception.
   * While performing pose transfer, it’s crucial to compare the output against a reference (either the target pose or the original image) to ensure the transfer's accuracy and quality. SSIM allows for this comparison in a way that reflects human perception, making it an excellent tool for evaluating the success of pose transfers in maintaining the appearance of the subject across transformations.
   * Pose transfer inherently involves significant structural changes to the subject in the image. SSIM's emphasis on structural information makes it particularly suitable for assessing how well these changes preserve the naturalness and coherence of the image compared to the reference. It helps in quantifying the effectiveness of ControlNet in transferring poses without introducing distortions or artifacts that could degrade the image quality.

**Implications of a High SSIM:**
 * Original Appearance is preserved.
 * Pose alignment but also maintain high perceptual quality.
 * Better user satisfaction with the pose-transferred images.
   
**This is applicable only for the Identity-aware Pose Retargeting** since we attempt to preserve the original appearance of the source image. For the Text Conditioned Image Editing, the judgement is more qualitative about the relevance to prompt and fidelity of editing.

## Results

### Final Pipeline Results
<p align="center"> 
<img width="1280" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/f20325e9-6a77-477d-9da3-b100d0aafa40">
</p>

### Comparison with Baseline
The existing methods we compare with are Stable Diffusion ControlNet Reference Pipeline and IP-adapter for ControlNet.

<p align="center"> 
<img width="1380" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/57be040f-b27e-41dd-86b6-69c3e14d8bfc">
</p>

The results show that our final architecture outperforms existing baselines, it achieves accurate pose transfer whilst retaining the identity of the source image greatly.

### Performance
The metric is mainly applicable to Identity-aware Pose Retargeting where we try to compare the generated image with the source to ensure the identity of the original image is preserved considerably. 
<p align="center"> 
<img alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/80c27729-f000-4b63-8f22-63dc3f386993">
</p>

The reported SSIM values for the above poses on a source image from the Tiktok Dataset are:
```
SSIM for pose1: 0.648
SSIM for pose2: 0.579
SSIM for pose3: 0.712
```
The values are close to 1, which shows that are model does a good job in preserving the identity of the original image while transfering the pose.

## Discussion
* We propose a novel image editing method called `PoseMaster Fusion` which edits images by transfering poses and making edits conditioned by text prompts. To that end, we utilize a pre-trained text-to-image diffusion model, optimize and finetune it to fit the image better and interpolate between the embedding representing the image and the target text embedding, obtaining a semantically meaningful mixture of them thus resulting in accurately edited images. These are then fed into a Appearance Disentangled Pose Control Module along with the pose that is to be transfered to seamlessly incorporate the pose and facial expression while enabling the generation of consistent inages without any further fine-tuning. This Text Conditioned Subject Driven Image Editing Mechanism is non rigid, preserves appearance, quick and accurate.
* Through this project we learnt a lot about Diffusion Models, their architecture, how to modify their architecture for spatial manipulations or for enhanced attention.
* We learnt about DDIM Sampling and Stochastic DDPM schemes for effectively finetuning these Diffusion Models.
* Through our exploration, we learnt more about current research trends in the field of Computer Vision and speficially generative models. We surveyed a lot of different models and analysed their pros and cons such as T2I Adapters and Older SD Versions.
* Our future work may focus on performing full body pose transfer as most datasets we used had more data related to just the upper body and very few samples for the whole frame. We also plan to extend our framework to improve the fidelity of image editing by enhancing the Text-conditioned Image editing to perform well for complex and longer prompts by training it more.
* Minimizing artifacts and inconsistencies in the generated images, especially in complex or uncommon poses, proved difficult. Improving the model's understanding and handling of background and foreground elements to alleviate these issues will be a focus of our future optimization efforts.

## Challenges
* We were often limited by the access to limited computation resources for limited time which significantly hindered our pace and progress. The computational demands of fine-tuning our models, especially given the complexity and scale of the data involved, were substantial. Managing these demands while ensuring efficient use of resources required meticulous planning.
* We also felt there was a dearth of good human pose datasets. A dataset with a lot of images of the same person in different poses would be ideal for pose transfer tasks, such datasets were scarce or not diverse enough, impacting our model's generalizability. Using Youtube and TikTok Video Frames, which is the current solution, results in poor quality of results due to occlusions and blur.
* Achieving a balance between maintaining the original image's identity and characteristics while accurately transferring new poses was challenging. Ensuring that the model could handle various poses without losing the subject's essence required careful tuning of model parameters and innovative use of attention mechanisms.
* Integrating ControlNet for pose transfer with existing frameworks like Stable Diffusion presented a significant technical challenge. Ensuring compatibility, optimizing data flow, and maintaining performance across different components of our pipeline required a deep understanding of each model's inner workings and considerable trial and error.
  
**Reflection**

  These challenges, while daunting, spurred innovation and collaboration within our team. They pushed us to explore novel solutions, delve deeper into the theoretical underpinnings of our work, and refine our approach through iterative testing and learning. The project's hurdles have prepared us better for future endeavors in the field of AI and image synthesis, highlighting the importance of resilience, creativity, and rigorous scientific inquiry in overcoming obstacles.
  If we were to start over, we would probably start sooner and collaborate more often to ensure steady progress over the project. We would also make better choices of datasets, our prepare our own dataset with a collection of clear, diverse stock images.
  


## Team Member Contribution

| Tasks | Member |
|:------|:-------|
|Exploration of Methods and Experiment Setup| All|
| Design an Appearance Control Module using a pretrained SD U-Net and auxiliary U-Net. | Shubham |
| Finetune ControlNet with the Appearance Control Module, freezing the Auxiliary U-Net to preserve appearance, thus creating an Appearance Disentangled Pose Control Module. | Poojitha |
| Understand and incorporate text-guided enhancements via the InstructPix2Pix module of SD. | Aparna |
| Evaluate the entire model's results based on the decided metrics. | Shubham |
| Examine the model's generalizability on other datasets | Aparna and Poojitha|
| Update the website and maintain the repository. | All |


## References

[1] M. Cao, X. Wang, Z. Qi, Y. Shan, X. Qie, and Y. Zheng, “MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing,” 2023. [Available Online](https://arxiv.org/abs/2304.08465).

[2] D. Chang, Y. Shi, Q. Gao, J. Fu, H. Xu, G. Song, Q. Yan, Y. Zhu, X. Yang, and M. Soleymani, “MagicPose: Realistic Human Poses and Facial Expressions Retargeting with Identity-aware Diffusion,” 2023. [Available Online](https://arxiv.org/abs/2311.12052).

[3] W. Li, X. Xu, J. Liu, and X. Xiao, “UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion,” 2024. [Available Online](https://arxiv.org/abs/2401.13388).

[4] D. Li, J. Li, and S. C. H. Hoi, “BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing,” 2023. [Available Online](https://ar5iv.labs.arxiv.org/html/2305.14720).

[5] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” 2020. [Available Online](https://arxiv.org/abs/2006.11239).

[6] T. Brooks, A. Holynski, and A. A. Efros, “InstructPix2Pix: Learning to Follow Image Editing Instructions,” 2022. [Available Online](https://arxiv.org/abs/2211.09800).

### Code References

[7] [Stable Diffusion ControlNet Reference Pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#stable-diffusion-controlnet-reference)

[8] [IP-Adapter for ControlNet](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter_controlnet_demo_new.ipynb)

[9] [Text Guided Image Editing](https://github.com/justinpinkney/stable-diffusion)

[10] [Identity Aware Pose Retargeting](https://github.com/Boese0601/MagicDance/tree/main)
