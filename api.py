API_PNG_INFO: str = "sdapi/v1/png-info"
API_TXT2IMG: str = "sdapi/v1/txt2img"
API_IMG2IMG: str = "sdapi/v1/img2img"


API_CONTROLNET_MODEL_LIST: str = "controlnet/model_list"
API_CONTROLNET_MODULE_LIST: str = "controlnet/module_list"
CONTROLNET_MODULE_KEY: str = "module_list"  # used to extract the list obj in the response json
CONTROLNET_MODEL_KEY: str = "model_list"
API_CONTROLNET_DETECT: str = "controlnet/detect"
CONTROLNET_KEY = "controlnet"


ALWAYSON_SCRIPTS_KEY = "alwayson_scripts"
ARGS_KEY = "args"  # used to parse always on scripts
INIT_IMAGES_KEY = "init_images"  # used in img2img payload making
IMAGE_KEY = "image"  # used in png-info payload making
IMAGES_KEY = "images"  # used in txt2img payload making
PNG_INFO_KEY = "info"
