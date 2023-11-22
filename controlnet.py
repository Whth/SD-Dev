import warnings
from typing import Dict, List, Optional

from aiohttp import ClientConnectorError, ClientSession
from pydantic import BaseModel

from .api import (
    API_CONTROLNET_MODEL_LIST,
    API_CONTROLNET_MODULE_LIST,
    API_CONTROLNET_DETECT,
    CONTROLNET_KEY,
    CONTROLNET_MODEL_KEY,
    CONTROLNET_MODULE_KEY,
    IMAGES_KEY,
    alwayson_scripts_pyload_wrapper,
)


class ControlNetUnit(BaseModel):
    """
    containing control net parsers
    detailed api wiki see
    https://github.com/Mikubill/sd-webui-controlnet/wiki/API#integrating-sdapiv12img
    """

    input_image: str
    module: str
    model: str
    resize_mode: int = 1
    """
    resize_mode : how to resize the input image so as to fit the output resolution of the generation.
     defaults to "Scale to Fit (Inner Fit)". Accepted values:
    0 or "Just Resize" : simply resize the image to the target width/height
    1 or "Scale to Fit (Inner Fit)" : scale and crop to fit smallest dimension. preserves proportions.
    2 or "Envelope (Outer Fit)" : scale to fit largest dimension. preserves proportions.
    """
    processor_res: int = 512
    weight: float = 1.0
    guidance: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    control_mode: int = 0
    """
    control_mode : see the related issue for usage. defaults to 0. Accepted values:
    0 or "Balanced" : balanced, no preference between prompt and control model
    1 or "My prompt is more important" : the prompt has more impact than the model
    2 or "ControlNet is more important" : the controlnet model has more impact than the prompt
    """


class ControlNetDetect(BaseModel):
    """
    Represents a controlnet detection configuration.

    Attributes:
        controlnet_module (str): The module for controlnet detection.
        controlnet_input_images (List[str]): The list of input images for detection.
        controlnet_processor_res (int): The processor resolution for detection. Default is 512.
        controlnet_threshold_a (int): The threshold A value for detection. Default is 64.
        controlnet_threshold_b (int): The threshold B value for detection. Default is 64.
    """

    controlnet_module: str
    controlnet_input_images: List[str]
    controlnet_processor_res: int = 512
    controlnet_threshold_a: int = 64
    controlnet_threshold_b: int = 64


class Controlnet(object):
    """
    Represents a controlnet client for performing various operations.

    Attributes:
        _host_url (str): The host URL for the controlnet API.
        _model_list (List[str]): The list of available models.
        _module_list (List[str]): The list of available modules.
    """

    @property
    def models(self) -> List[str]:
        """
        Returns a list of strings representing the available models.
        """
        return self._model_list

    @property
    def modules(self) -> List[str]:
        """
        Returns the list of modules.

        :return: A list of strings representing the modules.
        :rtype: List[str]
        """
        return self._module_list

    @classmethod
    async def __async_get(cls, url: str, session: Optional[ClientSession] = None) -> Dict:
        """
        An asynchronous class method that makes a GET request to the specified URL using the provided session. If no session is provided, a new session is created. Returns the JSON response as a dictionary.

        Parameters:
            url (str): The URL to make the GET request to.
            session (Optional[ClientSession]): An optional session to use for the GET request. If None, a new session will be created.

        Returns:
            Dict: The JSON response as a dictionary.
        """

        if session:
            response = await session.get(url=url)
        else:
            async with ClientSession() as session:
                response = await session.get(url=url)
        return await response.json()

    @classmethod
    async def __async_post(cls, url: str, payload: Dict, session: Optional[ClientSession] = None) -> Dict:
        """
        Sends a POST request to the specified URL with the given payload asynchronously.

        Args:
            url (str): The URL to send the request to.
            payload (Dict): The payload to include in the request.
            session (Optional[ClientSession], optional): The `ClientSession` object to use for the request.
                                                       If not provided, a new session will be created.

        Returns:
            Dict: The response from the request as a JSON object.
        """
        if session:
            response = await session.post(url=url, json=payload)
        else:
            async with ClientSession() as session:
                response = await session.post(url=url, json=payload)
        return await response.json()

    def __init__(self, host_url: str):
        """
        Initializes a new instance of the Controlnet class with the specified host URL.

        Args:
            host_url (str): The host URL for the controlnet API.
        """
        self._host_url: str = host_url

        self._model_list: List[str] = []
        self._module_list: List[str] = []

    async def fetch_resources(self, session: Optional[ClientSession] = None):
        """
        Asynchronously fetches the model and module lists from the API and
        stores them in the corresponding attributes.
        """
        try:
            fetch_session = session or ClientSession(base_url=self._host_url)
            self._model_list: List[str] = await self.get_model_list(session=fetch_session)
            self._module_list: List[str] = await self.get_module_list(session=fetch_session)
        except ClientConnectorError:
            pass

    async def get_model_list(self, session: Optional[ClientSession] = None) -> List[str]:
        """
        Retrieves a list of models from the API.

        Args:
            session: An optional ClientSession object for making the API request.

        Returns:
            A list of model names.
        """

        if session:
            # Make API request with session
            response = await self.__async_get(API_CONTROLNET_MODEL_LIST, session=session)
        else:
            # Make API request without session
            response = await self.__async_get(f"{self._host_url}{API_CONTROLNET_MODEL_LIST}")

        models: List[str] = response.get(CONTROLNET_MODEL_KEY)

        # Remove hash suffix from model names
        removed_suffix = [model.split(" ")[0] for model in models]

        return removed_suffix

    async def get_module_list(self, session: Optional[ClientSession] = None) -> List[str]:
        """
        Retrieves a list of modules from the controlnet API.

        Returns:
            A list of module names.
        """
        if session:
            # Make API request with session
            response = await self.__async_get(API_CONTROLNET_MODULE_LIST, session=session)
        else:
            # Make API request without session
            response = await self.__async_get(f"{self._host_url}{API_CONTROLNET_MODULE_LIST}")
        return response.get(CONTROLNET_MODULE_KEY)

    async def detect(self, payload: ControlNetDetect) -> List[str]:
        """
        Asynchronously detects the given payload.

        Args:
            payload (ControlNetDetect): The payload to be detected.

        Returns:
            List[str]: A list of detected images, in base64 format
        """

        if payload.controlnet_module not in self._module_list:
            raise KeyError("invalid controlnet module,please check")
        return (await self.__async_post(f"{self._host_url}{API_CONTROLNET_DETECT}", payload=payload.dict())).get(
            IMAGES_KEY
        )

    async def make_cn_payload_safe(self, units: List[ControlNetUnit]) -> Dict:
        """
        Generates a safe payload for ControlNet units.

        This method takes a list of ControlNetUnit objects and filters out the units
        that are not present in the module and model lists. It returns a dictionary
        containing the filtered units.

        Parameters:
            units (List[ControlNetUnit]): A list of ControlNetUnit objects to be filtered.

        Returns:
            Dict: A dictionary containing the filtered ControlNetUnit objects.

        Raises:
            ValueError: If all the units in the list are invalid.

        Warnings:
            If the number of received units is not equal to the number of filtered units.

        """
        passed_units: List[ControlNetUnit] = []
        for unit in units:
            if unit.module in self._module_list and unit.model in self._model_list:
                passed_units.append(unit)
        if len(passed_units) == 0:
            raise ValueError("All units are invalid")
        elif len(passed_units) != len(units):
            warnings.warn(f"Received {len(units)} cn_units, but only {len(passed_units)} have passed the check")
        return make_cn_payload(passed_units)


@alwayson_scripts_pyload_wrapper(CONTROLNET_KEY)
def make_cn_payload(units: List[ControlNetUnit]) -> List[Dict]:
    """
    Generate a payload dictionary from a list of ControlNetUnit objects.

    Parameters:
        units (List[ControlNetUnit]): A list of ControlNetUnit objects.

    Returns:
        Dict: A dictionary containing the payload with the ControlNetUnit objects.
    """
    unit_seq: List[Dict] = []
    for unit in units:
        unit_seq.append(unit.dict())
    return unit_seq
