from .nodes_wan import *

NODE_CLASS_MAPPINGS = {
    "WanImageToVideoTiledVAE": WanImageToVideoTiledVAE,
    "WanFunControlToVideoTiledVAE": WanFunControlToVideoTiledVAE,
    "WanFirstLastFrameToVideoTiledVAE": WanFirstLastFrameToVideoTiledVAE,
    "WanFunInpaintToVideoTiledVAE": WanFunInpaintToVideoTiledVAE,
    "WanVaceToVideoTiledVAE": WanVaceToVideoTiledVAE,
    "WanCameraImageToVideoTiledVAE":WanCameraImageToVideoTiledVAE,
    "WanPhantomSubjectToVideoTiledVAE": WanPhantomSubjectToVideoTiledVAE,
    "WanTrackToVideoTiledVAE": WanTrackToVideoTiledVAE,
    "Wan22ImageToVideoLatentTiledVAE":Wan22ImageToVideoLatentTiledVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanImageToVideoTiledVAE": "WanImageToVideo (Tiled VAE encode)",
    "WanFunControlToVideoTiledVAE": "WanFunControlToVideo (Tiled VAE encode)",
    "WanFirstLastFrameToVideoTiledVAE": "WanFirstLastFrameToVideo (Tiled VAE encode)",
    "WanFunInpaintToVideoTiledVAE": "WanFunInpaintToVideo (Tiled VAE encode)",
    "WanVaceToVideoTiledVAE": "WanVaceToVideo (Tiled VAE encode)",
    "WanCameraImageToVideoTiledVAE": "WanCameraImageToVideo (Tiled VAE encode)",
    "WanPhantomSubjectToVideoTiledVAE": "WanPhantomSubjectToVideo (Tiled VAE encode)",
    "WanTrackToVideoTiledVAE": "WanTrackToVideo (Tiled VAE encode)",
    "Wan22ImageToVideoLatentTiledVAE":"Wan22ImageToVideoLatent (Tiled VAE encode)",
}
