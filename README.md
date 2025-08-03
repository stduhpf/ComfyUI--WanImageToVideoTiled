# WanImageToVideoTiledVAE for ComfyUI

This is a set of custom nodes for ComfyUI that replaces nodes like WanImageToVideo but using a Tiled VAE approach to reduce VRAM requirements.

## Description

WanImageToVideo (Tiled VAE encode) is designed to be a drop-in replacement for the WanImageToVideo node in ComfyUI. By using a tiled VAE, it significantly reduces the VRAM needed, making it more accessible for users with limited resources. I found it very useful for [ComfyUI-zluda](https://github.com/patientx/ComfyUI-Zluda), since the VAE is particularly VRAM-hungry there.

## Installation

To install this node, follow these steps:

1. Clone this repository into your ComfyUI custom nodes directory.
2. Restart ComfyUI to load the new node.

```bash
git clone https://github.com/stduhpf/ComfyUI--WanImageToVideoTiled.git /path/to/ComfyUI/custom_nodes/WanImageToVideoTiledVAE
```

## Usage

Once the extension is installed, you can just replace the original node with the one from this extension.

## New Inputs (from the `VAE encode (Tiled)` node)

| Input | Type | Description |
|-------|------|-------------|
| tile_size | INT | Size of the tiles, default is 512. |
| overlap | INT | Overlap between tiles, default is 64. |
| temporal_size | INT | Amount of frames to encode at a time, default is 64. |
| temporal_overlap | INT | Amount of frames to overlap, default is 8. |

## Included nodes:

Here are the nodes included in this extension, checkmark means the node has been succesfuly tested. Unchecked ones should work too, if you try these nodes and want to give feedback, you can open an issue or a discussion.

- [x] WanImageToVideo (Tiled VAE encode) 
- [x] WanFunControlToVideo (Tiled VAE encode)
- [x] WanFirstLastFrameToVideo (Tiled VAE encode)
- [ ] WanFunInpaintToVideo (Tiled VAE encode)
- [x] WanVaceToVideo (Tiled VAE encode)
- [ ] WanCameraImageToVideo (Tiled VAE encode)
- [ ] WanPhantomSubjectToVideo (Tiled VAE encode)
- [ ] WanTrackToVideo (Tiled VAE encode)
- [x] Wan22ImageToVideoLatent (Tiled VAE encode)

## License

This project mostly contains code copy-pasted from ComfyUI, which is licenced under GPL3.0. Therefore it is also licenced under GPL 3.0. (see LICENCE file for more details)
