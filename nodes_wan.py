from comfy.sd import VAE
# import math
from comfy_extras.nodes_wan import pad_pts, parse_json_tracks, patch_motion, process_tracks
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
# import comfy.clip_vision
# import json
import numpy as np
# from typing import Tuple


class WanImageToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),

                             "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                             "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                             "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                             "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae:VAE, width, height, length, batch_size,  tile_size, overlap, temporal_size=64, temporal_overlap=8, start_image=None, clip_vision_output=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode_tiled(image[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

class WanFunControlToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             
                             "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                             "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                             "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                             "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "control_video": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size,  tile_size, overlap, temporal_size=64, temporal_overlap=8, start_image=None, clip_vision_output=None, control_video=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode_tiled(start_image[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            concat_latent[:,16:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode_tiled(control_video[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            concat_latent[:,:16,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

class WanFirstLastFrameToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),

                             "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                             "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                             "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                             "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {"clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                             "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size,  tile_size, overlap, temporal_size=64, temporal_overlap=8, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        concat_latent_image = vae.encode_tiled(image[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

class WanFunInpaintToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                                "negative": ("CONDITIONING", ),
                                "vae": ("VAE", ),
                                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),

                                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                                "start_image": ("IMAGE", ),
                                "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size,  tile_size, overlap, temporal_size=64, temporal_overlap=8, start_image=None, end_image=None, clip_vision_output=None):
        flfv = WanFirstLastFrameToVideoTiledVAE()
        return flfv.encode(positive, negative, vae, width, height, length, batch_size,tile_size, overlap,temporal_size=temporal_size,temporal_overlap=temporal_overlap,  start_image=start_image, end_image=end_image, clip_vision_start_image=clip_vision_output)


class WanVaceToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                                "negative": ("CONDITIONING", ),
                                "vae": ("VAE", ),
                                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),

                                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {"control_video": ("IMAGE", ),
                                "control_masks": ("MASK", ),
                                "reference_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size, strength,  tile_size, overlap, temporal_size=64, temporal_overlap=8, control_video=None, control_masks=None, reference_image=None):
        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            reference_image = vae.encode_tiled(reference_image[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode_tiled(inactive[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
        reactive = vae.encode_tiled(reactive[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent, trim_latent)

class WanCameraImageToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                                "negative": ("CONDITIONING", ),
                                "vae": ("VAE", ),
                                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                            
                                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                                "start_image": ("IMAGE", ),
                                "camera_conditions": ("WAN_CAMERA_EMBEDDING", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size,  tile_size, overlap, temporal_size=64, temporal_overlap=8, start_image=None, clip_vision_output=None, camera_conditions=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode_tiled(start_image[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            concat_latent[:,:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if camera_conditions is not None:
            positive = node_helpers.conditioning_set_values(positive, {'camera_conditions': camera_conditions})
            negative = node_helpers.conditioning_set_values(negative, {'camera_conditions': camera_conditions})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

class WanPhantomSubjectToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                                "negative": ("CONDITIONING", ),
                                "vae": ("VAE", ),
                                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                            
                                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {"images": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative_text", "negative_img_text", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, tile_size, overlap, temporal_size, temporal_overlap, images):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        cond2 = negative
        if images is not None:
            images = comfy.utils.common_upscale(images[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_images = []
            for i in images:
                latent_images += [vae.encode_tiled(i.unsqueeze(0)[:, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)]
            concat_latent_image = torch.cat(latent_images, dim=2)

            positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
            cond2 = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
            negative = node_helpers.conditioning_set_values(negative, {"time_dim_concat": comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, cond2, negative, out_latent)


class WanTrackToVideoTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "tracks": ("STRING", {"multiline": True, "default": "[]"}),
                    "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                    "temperature": ("FLOAT", {"default": 220.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                    "topk": ("INT", {"default": 2, "min": 1, "max": 10}),
                    "start_image": ("IMAGE", ),

                    "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                    "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                    "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {
                    "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, tracks, width, height, length, batch_size,
                temperature, topk,  tile_size, overlap, temporal_size=64, temporal_overlap=8, start_image=None, clip_vision_output=None):

        tracks_data = parse_json_tracks(tracks)

        if not tracks_data:
            return WanImageToVideoTiledVAE().encode(positive, negative, vae, width, height, length, batch_size, tile_size, overlap, temporal_size=temporal_size, temporal_overlap=temporal_overlap, start_image=start_image, clip_vision_output=clip_vision_output)

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                            device=comfy.model_management.intermediate_device())

        if isinstance(tracks_data[0][0], dict):
            tracks_data = [tracks_data]

        processed_tracks = []
        for batch in tracks_data:
            arrs = []
            for track in batch:
                pts = pad_pts(track)
                arrs.append(pts)

            tracks_np = np.stack(arrs, axis=0)
            processed_tracks.append(process_tracks(tracks_np, (width, height), length - 1).unsqueeze(0))

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:batch_size].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            videos = torch.ones((start_image.shape[0], length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            for i in range(start_image.shape[0]):
                videos[i, 0] = start_image[i]

            latent_videos = []
            videos = comfy.utils.resize_to_batch_size(videos, batch_size)
            for i in range(batch_size):
                latent_videos += [vae.encode_tiled(videos[i, :, :, :, :3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)]
            y = torch.cat(latent_videos, dim=0)

            # Scale latent since patch_motion is non-linear
            y = comfy.latent_formats.Wan21().process_in(y)

            processed_tracks = comfy.utils.resize_list_to_batch_size(processed_tracks, batch_size)
            res = patch_motion(
                processed_tracks, y, temperature=temperature, topk=topk, vae_divide=(4, 16) # type: ignore
            )

            mask, concat_latent_image = res
            concat_latent_image = comfy.latent_formats.Wan21().process_out(concat_latent_image)
            mask = -mask + 1.0  # Invert mask to match expected format
            positive = node_helpers.conditioning_set_values(positive,
                                                            {"concat_mask": mask,
                                                            "concat_latent_image": concat_latent_image})
            negative = node_helpers.conditioning_set_values(negative,
                                                            {"concat_mask": mask,
                                                            "concat_latent_image": concat_latent_image})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class Wan22ImageToVideoLatentTiledVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE", ),
                            "width": ("INT", {"default": 1280, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                            "height": ("INT", {"default": 704, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                            "length": ("INT", {"default": 49, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),

                            "tile_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                            "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                            "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Amount of frames to encode at a time."}),
                            "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Amount of frames to overlap."})
                },
                "optional": {"start_image": ("IMAGE", ),
                }}


    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "conditioning/inpaint"

    def encode(self, vae, width, height, length, batch_size,  tile_size, overlap, temporal_size=64, temporal_overlap=8, start_image=None):
        latent = torch.zeros([1, 48, ((length - 1) // 4) + 1, height // 16, width // 16], device=comfy.model_management.intermediate_device())

        if start_image is None:
            out_latent = {}
            out_latent["samples"] = latent
            return (out_latent,)

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 4) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_temp = vae.encode_tiled(start_image, tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            latent[:, :, :latent_temp.shape[-3]] = latent_temp
            mask[:, :, :latent_temp.shape[-3]] *= 0.0

        out_latent = {}
        latent_format = comfy.latent_formats.Wan22()
        latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
        out_latent["samples"] = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))
        return (out_latent,)
