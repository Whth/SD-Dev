from typing import Callable, Any

API_PNG_INFO: str = "/sdapi/v1/png-info"
API_TXT2IMG: str = "/sdapi/v1/txt2img"
API_IMG2IMG: str = "/sdapi/v1/img2img"
API_MODELS: str = "/sdapi/v1/sd-models"
API_LORAS: str = "/sdapi/v1/loras"
API_GET_CONFIG: str = "/sdapi/v1/options"
API_GET_UPSCALERS: str = "/sdapi/v1/upscalers"

API_INTERROGATE: str = "/sdapi/v1/interrogate"
API_INTERRUPT: str = "/sdapi/v1/interrupt"
API_SAMPLERS: str = "/sdapi/v1/samplers"
API_MODULES: str = "/sdapi/v1/sd-modules"
API_CONTROLNET_MODEL_LIST: str = "/controlnet/model_list"
API_CONTROLNET_MODULE_LIST: str = "/controlnet/module_list"
CONTROLNET_MODULE_KEY: str = "module_list"  # used to extract the list obj in the response json
CONTROLNET_MODEL_KEY: str = "model_list"
API_CONTROLNET_DETECT: str = "/controlnet/detect"
CONTROLNET_KEY = "controlnet"

ALWAYSON_SCRIPTS_KEY = "alwayson_scripts"
ARGS_KEY = "args"  # used to parse always on scripts
INIT_IMAGES_KEY = "init_images"  # used in img2img payload making
IMAGE_KEY = "image"  # used in png-info payload making
IMAGES_KEY = "images"  # used in txt2img payload making
PNG_INFO_KEY = "info"


def alwayson_scripts_pyload_wrapper(script_name: str) -> Callable[[Callable], Callable]:
    """
    Decorator that wraps a function and returns a dictionary containing the function's result.

    Parameters:
        script_name (str): The name of the script.

    Returns:
        Callable: A wrapper function that returns a dictionary with the function's result.
    """

    def alwayson_scripts_pyload(fn: Callable[..., Any]) -> Callable[..., dict]:
        """
        Decorator that wraps a function and returns a dictionary containing the function's result.

        Parameters:
            fn (Callable): The function to be wrapped.

        Returns:
            Callable: A wrapper function that returns a dictionary with the function's result.
        """

        def inner(*args, **kwargs) -> dict:
            """
            Wrapper function that returns a dictionary containing the result of the wrapped function.

            Parameters:
                *args: Positional arguments to be passed to the wrapped function.
                **kwargs: Keyword arguments to be passed to the wrapped function.

            Returns:
                dict: A dictionary containing the result of the wrapped function.
            """
            return {script_name: {ARGS_KEY: fn(*args, **kwargs)}}

        return inner

    return alwayson_scripts_pyload


def merge_payloads(*payloads: dict) -> dict:
    """
    Merge multiple payloads into a single dictionary.

    Args:
        *payloads (dict): Multiple dictionaries to be merged.

    Returns:
        dict: A dictionary containing the merged payloads.
    """
    temp = {}
    for payload in payloads:
        temp.update(payload) if payload else None
    return temp
