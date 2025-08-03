from .nodes_wan import WanImageToVideoTiledVAE,WanFirstLastFrameToVideoTiledVAE

NODE_CLASS_MAPPINGS = {
    "WanImageToVideoTiledVAE": WanImageToVideoTiledVAE,
    "WanFirstLastFrameToVideoTiledVAE": WanFirstLastFrameToVideoTiledVAE
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanImageToVideoTiledVAE": "WanImageToVideo (Tiled VAE encode)",
    "WanFirstLastFrameToVideoTiledVAE": "WanFirstLastFrameToVideo (Tiled VAE encode)",
}
