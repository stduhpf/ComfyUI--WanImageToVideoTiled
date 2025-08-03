# WanImageToVideoTiledVAE for ComfyUI

This is a custom node for ComfyUI that replaces the WanImageToVideo node but uses a Tiled VAE approach to reduce VRAM requirements.

## Description

WanImageToVideoTiledVAE is designed to be a drop-in replacement for the WanImageToVideo node in ComfyUI. By using a tiled VAE, it significantly reduces the VRAM needed, making it more accessible for users with limited resources. I found it very useful for [ComfyUI-zluda](https://github.com/patientx/ComfyUI-Zluda), since the VAE is particularly VRAM-hungry there.

## Installation

To install this node, follow these steps:

1. Clone this repository into your ComfyUI custom nodes directory.
2. Restart ComfyUI to load the new node.

```bash
git clone https://github.com/stduhpf/ComfyUI--WanImageToVideoTiled.git /path/to/ComfyUI/custom_nodes/WanImageToVideoTiledVAE
```

## Usage

To use this node in ComfyUI:

1. Add the `WanImageToVideoTiledVAE` node to your workflow.
2. Connect the required inputs (positive and negative conditioning, VAE model, start image).
3. Optionally, connect a start image or clip vision output.
4. Run the workflow.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| positive | CONDITIONING | Positive conditioning input. |
| negative | CONDITIONING | Negative conditioning input. |
| vae | VAE | The VAE model to use. |
| width | INT | Width of the video, default is 832. |
| height | INT | Height of the video, default is 480. |
| length | INT | Number of video frames, default is 81. |
| batch_size | INT | Batch size, default is 1. |
| tile_size | INT | Size of the tiles, default is 512. |
| overlap | INT | Overlap between tiles, default is 64. |
| temporal_size | INT | Amount of frames to encode at a time, default is 64. |
| temporal_overlap | INT | Amount of frames to overlap, default is 8. |
| start_image | IMAGE (optional) | Optional start image. |
| clip_vision_output | CLIP_VISION_OUTPUT (optional) | Optional clip vision output. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| latent | LATENT | Empty latents with start_image encoded as first frame. |
| positive | CONDITIONING | Processed positive conditioning. |
| negative | CONDITIONING | Processed negative conditioning. |

## License

This project mostly contains code copy-pasted from ComfyUI, which is licenced under GPL3.0. Therefore it is also licenced under GPL 3.0. (see LICENCE file for more details)
