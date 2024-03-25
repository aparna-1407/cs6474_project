---
layout: default
---
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






