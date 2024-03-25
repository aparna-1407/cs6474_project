---
layout: default
---
# PoseMaster Fusion: Transforming Images by capturing Pose Dynamics
## Introduction
<p align="justify">
PoseMaster Fusion is a diffusion based model for 2D human pose retargeting. Specifically, given a reference image, the model aims to generate a set of new images of the person by controlling the poses and other features such as the style while keeping the identity unchanged. At its core, this project seeks to revolutionize the way we interact with and manipulate images, providing users with unprecedented control over the pose dynamics of their digital images. By leveraging the ability of control nets to understand and map poses from reference images or doodles, and harmonizing this with the generative capabilities of stable diffusion models, Pose Master Fusion attempts to synthesize images disentangling appearance from pose and style. 
<p align="center">
  <img width="405" alt="image" src="https://github.com/aparna-1407/cs6474_project/assets/93538009/f22435b9-2d74-45f6-a653-9717402bb401">
</p>
  
This image can better explain what this model aims to accomplish- given a reference pose that can either be extracted from a doodle or a reference image, the model aims to translate that pose into the source image provided. PoseMaster Fusion allows you to control style in addition to just poses using text prompts to influence the style of your generated image.

<p align="center">
<img width="378" alt="image" src="https://github.com/aparna-1407/cs6474_project/assets/93538009/32fcffa9-6577-4d79-a40e-5f878a7b7bfe">
</p>

### Why is there a need for such a model?

This exploration is not just a technical endeavor but a response to the growing demand for more intuitive, powerful tools in digital art and media production. As content creation increasingly grows, there's a palpable need for technologies that can simplify complex processes like pose manipulation and style trasnfer, making them quick, simple and accessible. The ability to seamlessly transfer poses from one image to another, with minimal effort and high fidelity such as from doodles and text prompts, opens up new avenues for creativity and efficiency in digital content creation. While translating poses from reference images using diffusion models is a growing area pf research, we through PoseMaster Fusion attempt to extend pose transfer from using just reference images to even include doodles from which poses can be extracted and also envision to include a module that allows enhancements to style via text prompts.

The challenges in creating a cohesive system that can accurately interpret human poses, translate them into a different context while retaining the original image's essence, and produce results that are both visually stunning and contextually appropriate are immense. Through PoseMaster Fusion, we attempt to create a model that generalizes well and enables robust appearance control while leveraging the prior knowledge of Stable Diffusion models and ControlNet.
</p>

## Previous Work
### Controlled Image Synthesis
<p>
[1] **"MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing" by Cao et al.** propose to change pose, view, structures, and non-rigid variances of the source image while maintaining the characteristics, texture, and identity by converting existing self attention in diffusion models into mutual self-attention, so that it can query correlated local contents and textures from source images for consistency. To further alleviate the query confusion between foreground and background, they propose a mask-guided mutual self-attention strategy, where the mask can be easily extracted from the cross-attention maps in diffusion models to separate foreground and background. Thye change the non-rigid attributes (e.g., changing object pose) using only text prompts. The pipeline proposed is that the user feeds an image prompt Ps to describe the source image, a modified target prompt Pt (changing just one word in the source prompt) and provides the source image. The mutual attention mechanism queries image content so that it can generate consistent images under the modified target prompt. The mutual attention and cross attention layer extracts informations about features relevant for the target prompt from its knowledge and synthesize a semantic layout. The denoising U-Net injects this information to the source retaining consistency because of the separation of foreground and background. This pipeline that is solely controlled by text prompts along with Stable Diffusion was observed to be ineffective, so T2I adapters were also integrated for more stable synthesis. The major disadvantages are being limited by the knowledge of stable diffusion to generate the target prompt and impact of artifacts or change in background or other inconsistencies with the target prompt. 

We believe we can improve our model by incorporating the idea of poses instead of attention maps, avoiding text guidance and focusing on just pose driven image synthesis, integrating ControlNet and T2I adapters as they are superior in image editing than just prompt guide image synthesis by Stable Diffusion and finetuning the model on our dataset to adapt to artifacts and changes in appearance and improving knowledge.
</p>
<p>
[2] **"MagicPose: Realistic Human Poses and Facial Expressions Retargeting with Identity-aware Diffusion"** by Chang et al. proposes a diffusion model based 2d human pose and facial expression retargetting. To retain identity or consistency of source image they explored connecting the attention layers of diffusion U-Net to provide layer-by-layer attention guidance and retain the apperance of the source image as these layers are highly relevant to the appearance of the generated images.They thus pretain the Stable Diffusion U-Net along with a "Appearance Control Module" that has self attention layers, the key-value pairs are connected together and attention is calculated. ControlNet copies the encoder and middle blocks of SD-UNet, whose output feature maps are added to the decoder of SD-UNet to realize pose control while retaining the appearance. To enhance the entanglement, they fine-tune the Pose ControlNet jointly with our Appearance Control Module this provides enhanced results.

We believe we can improve our model by incorporating the appearance control module and finetuning our ControlNet with it.
</p>


