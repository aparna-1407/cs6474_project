---
layout: default
---
# PoseMaster Fusion: Transforming Images by capturing Pose Dynamics
## Contents
1. [Introduction](#introduction)
  * [Project Description](##project-description)
  * [Motivation](###why-is-there-a-need-for-such-a-model-?)
2. [Related Work](#related-work)
  * [Explaining Context](##explaining-context)
  * [Motivation](###why-is-there-a-need-for-such-a-model-?)
3. [Approach](#approach)
4. [Experiments and Results](#experiments-and-results)
5. [Future Steps](#future-steps)
6. [Contribution](#contribution)
7. [References](#references)

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
We believe we can improve our model by incorporating the idea of poses instead of attention maps, avoiding text guidance and focusing on just pose driven image synthesis, integrating ControlNet and T2I adapters as they are superior in image editing than just prompt guide image synthesis by Stable Diffusion and finetuning the model on our dataset to adapt to artifacts and changes in appearance and improving knowledge.
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
    
3. **Text Guided Enhancement Module** which consists of a text encoder which creates prompt embeddings and we combine this prompt embeddings used with subject/source embeddings (understanding the structure of the source) generated by the InstructPix2Pix model within stable diffusion. The finetuning diffusion process creates an interpolation of the embedding space to understand the prompt generated to describe the input image and the actual text prompt thereby creating an optimized text embedding to indicate parts of the image that should be changed and how. The cross attention layers of stable diffusion's cross attention layers uses the embeddings and masks regions to be edited and the prompt embeddings guides the editing. This will leverage the prior knowledge of stable diffusion and not require any finetuning.
#### Architecture
* This architecture has 3 modules- Embedding Optimization, Finetuning Diffusion Model and Interpolating Embedding space. We use the CLIP Text Encoder present within SD and SD V1.4 for this module as it is most stable.
  *  Given the target prompt A and the image B, the CLIP Encoder tries to determine a text embedding x describing image B and a text embedding y for prompt A with an objective of reducing the distance between x and y. The reason for determining close x and y is to minimize the number of edits to make to the image, or retain substantial amounts of the original image while editing only whatever is absolutely necessary. This is achieved by freezing the parameters of the generative diffusion model, and optimizing the CLIP Encoder using the denoising diffusion objective using a DDIM Sampler. While the denoising diffusion objective focuses on how to train a model to reverse the diffusion process accurately, the DDIM sampler offers a way to perform this reverse process more efficiently during inference. It leverages the trained model to produce samples in fewer steps by following an implicit trajectory in the noise space.
  * We fine-tune the diffusion models while freezing the CLIP Encoder to generate image C similar to text embedding x with an objective to reduce the reconstruction loss between image C and image B. This is to synchronise the diffusion model and the encoder to ensure that the generation process targets the areas of the image as necessary, in order to find a point that achieves both fidelity to the input image and target text alignment.
  * Since the generative diffusion model is now trained to fully recreate the input image B at the optimized embedding x, we use it to apply the desired edit A by advancing in the direction of the target text embedding y. We apply the base generative diffusion process SD 1.4 to generate an low-resolution edited image capturing features of A, which is then super-resolved using the fine-tuned SD backbone, conditioned on the target text y. This generative process outputs our final high resolution edited image D. The finetuned SD backbone retains the details of the original image and retargets regions specified by target embedding y. The new editted features of the image are supplied by the base SD1.4 as it has vast knowledge to generate the desired edit. 

> For Project Update-1, we have completed part of the Pose Control Module. We have identified how to set up the pipeline corrently and how ControlNet works best with Stable Diffusion to generate the right output. More details on the different approaches attempted are explained in the next sections. Only during the set up of our pipeline did we realize that ControlNet doesn't preserve appearance and based on further research arrived at the current architecture. So our future steps would be to build a fine-tuned ACM and then incorporate within the pipeline to create the ADPCM.

## Experiments and Results

(Work done for Project Update-1)
Following is a summary of all the approaches we tried, what we observed, and the challenges faced.

### Dataset

**[Deep Fashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)** 
<p> 
  We use the in-shop retrieval subset of the Deep Fashion dataset which consists of 52,712 high-resolution images of fashion models with diverse backgrounds,viewpoints,and lighting conditions. It contains the images of the same model available in different poses thus enabling the finetuning and training stages we plan to execute. We extract the skeletons using OpenPose. The model ID of the training and testing sets donot overlap. 
</p>

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
  
We will aim to address this issue with a improved pose-transfer model that we will develop in the next steps of our project.

<p align="center"> 
<img width="1280" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/592c672e-4192-4b24-836f-9d16272371b1">
<img width="1280" alt="image" src="https://github.com/aparna-1407/cs6476_project_team18/assets/93538009/8b5dca3e-ce3e-4914-9604-850838ceb8e1">
</p>




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
