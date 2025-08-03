from .nodes_wan import *

NODE_CLASS_MAPPINGS = {
    "WanImageToVideoTiledVAE": WanImageToVideoTiledVAE,
    "WanFunControlToVideoTiledVAE": WanFunControlToVideoTiledVAE,
    "WanFirstLastFrameToVideoTiledVAE": WanFirstLastFrameToVideoTiledVAE,
    "WanFunInpaintToVideoTiledVAE": WanFunInpaintToVideoTiledVAE,
    "WanVaceToVideoTiledVAE": WanVaceToVideoTiledVAE,
    "WanCameraImageToVideoTiledVAE":WanCameraImageToVideoTiledVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanImageToVideoTiledVAE": "WanImageToVideo (Tiled VAE encode)",
    "WanFunControlToVideoTiledVAE": "WanFunControlToVideo (Tiled VAE encode)",
    "WanFirstLastFrameToVideoTiledVAE": "WanFirstLastFrameToVideo (Tiled VAE encode)",
    "WanFunInpaintToVideoTiledVAE": "WanFunInpaintToVideo (Tiled VAE encode)",
    "WanVaceToVideoTiledVAE": "WanVaceToVideo (Tiled VAE encode)",
    "WanCameraImageToVideoTiledVAE": "WanCameraImageToVideo (Tiled VAE encode)",
}
