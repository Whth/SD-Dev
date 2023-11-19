import copy
import json
import pathlib
from collections import OrderedDict
from random import choice
from typing import List, Dict, Optional, Any

import aiohttp
from pydantic import BaseModel, Field, validator

from modules.file_manager import img_to_base64
from .adetailer import ADetailerArgs
from .api import (
    API_TXT2IMG,
    INIT_IMAGES_KEY,
    ALWAYSON_SCRIPTS_KEY,
    API_IMG2IMG,
    API_MODELS,
    API_LORAS,
    API_INTERROGATE,
    merge_payloads,
    API_GET_UPSCALERS,
)
from .controlnet import ControlNetUnit, make_cn_payload
from .parser import DiffusionParser, HiResParser, OverRideSettings, InterrogateParser, RefinerParser
from .utils import save_base64_img_with_hash, extract_png_from_payload


class RetrievedData(BaseModel):
    favorite: List[Dict]
    history: List[Dict]


class PersistentManager(BaseModel):
    class Config:
        allow_mutation = False
        validate_assignment = True

    save_path: str = Field(exclude=True)
    max_history: int = Field(default=20, exclude=True)
    current: Dict = Field(default_factory=dict, const=True, exclude=True)
    favorite: List[Dict] = Field(default_factory=list, const=True)
    history: List[Dict] = Field(default_factory=list, const=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.load() if pathlib.Path(self.save_path).exists() else None

    @validator("save_path")
    def validate_save_path(cls, v):
        pathlib.Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v

    def save(self):
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.dict(), f, indent=2, ensure_ascii=False)

    def load(self):
        self.favorite.clear()
        self.history.clear()
        with open(self.save_path, "r", encoding="utf-8") as f:
            loaded_data: RetrievedData = RetrievedData(**json.load(f))
            self.favorite.extend(loaded_data.favorite)
            self.history.extend(loaded_data.history)

    def payload_init(self):
        self.current.clear()

    def add_payload(self, payload: Dict):
        self.current.update(payload)

    def store(self, with_save=True):
        self.history.append(copy.deepcopy(self.current))
        self._prune_history()
        self.save() if with_save else None

    def add_favorite(self, with_save=True):
        """
        Adds the current item to the list of favorite items.

        Raises:
            ValueError: If there is no current item to add to the favorite list.

        Returns:
            None
        """
        if not self.current:
            raise ValueError("No current to add to favorite")
        if self.current in self.favorite:
            return

        self.favorite.append(copy.deepcopy(self.current))
        self.save() if with_save else None

    def _prune_history(self):
        for _ in range(len(self.history) - self.max_history):
            self.history.pop(0)


I2I_HISTORY_SAVE_PATH = "i2i_history.json"
T2I_HISTORY_SAVE_PATH = "t2i_history.json"


class StableDiffusionApp(BaseModel):
    """
    class that implements the basic diffusion api
    """

    host_url: str
    cache_dir: str
    output_dir: str
    img2img_params: Optional[PersistentManager]
    txt2img_params: Optional[PersistentManager]
    available_sd_models: List[str] = Field(default_factory=list, const=True)
    available_lora_models: List[str] = Field(default_factory=list, const=True)
    available_upscalers: List[str] = Field(default_factory=list, const=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.txt2img_params = PersistentManager(save_path=f"{self.cache_dir}/{T2I_HISTORY_SAVE_PATH}")
        self.img2img_params = PersistentManager(save_path=f"{self.cache_dir}/{I2I_HISTORY_SAVE_PATH}")

    def add_favorite_i(self) -> bool:
        self.img2img_params.add_favorite()
        return True

    def add_favorite_t(self) -> bool:
        self.txt2img_params.add_favorite()
        return True

    async def txt2img(
        self,
        diffusion_parameters: DiffusionParser = DiffusionParser(),
        hires_parameters: Optional[HiResParser] = None,
        refiner_parameters: Optional[RefinerParser] = None,
        controlnet_parameters: Optional[ControlNetUnit] = None,
        adetailer_parameters: Optional[ADetailerArgs] = None,
        override_settings: Optional[OverRideSettings] = None,
    ) -> List[str]:
        """
        Generate an image from a text using the specified parameters.

        Args:
            diffusion_parameters (DiffusionParser, optional): The parameters for the diffusion process.
                Defaults to DiffusionParser().
            hires_parameters (HiResParser, optional): The parameters for generating high-resolution images.
                Defaults to HiResParser().
            refiner_parameters (RefinerParser, optional): The parameters for refining the generated images.
                Defaults to None.
            controlnet_parameters (ControlNetUnit, optional): The parameters for controlling the net. Defaults to None.
            adetailer_parameters (ADetailerArgs, optional): The parameters for the adetailer. Defaults to None.
            override_settings (OverRideSettings, optional): The settings for overriding the default parameters.
                Defaults to OverRideSettings().

        Returns:
            List[str]: A list of paths to the generated images.
        """

        self.txt2img_params.payload_init()

        self.txt2img_params.add_payload(
            merge_payloads(
                diffusion_parameters.dict(),
                hires_parameters.dict(exclude_none=True) if hires_parameters else None,
                override_settings.dict() if override_settings else None,
                refiner_parameters.dict() if refiner_parameters else None,
            )
        )

        self.txt2img_params.add_payload(
            {
                ALWAYSON_SCRIPTS_KEY: merge_payloads(
                    adetailer_parameters.make_pyload() if adetailer_parameters else None,
                    make_cn_payload([controlnet_parameters]) if controlnet_parameters else None,
                )
            }
        )

        images_paths = await self._make_image_gen_request(self.txt2img_params.current, API_TXT2IMG)

        self.txt2img_params.store()

        return images_paths

    async def img2img(
        self,
        diffusion_parameters: DiffusionParser = DiffusionParser(),
        refiner_parameters: Optional[RefinerParser] = None,
        controlnet_parameters: Optional[ControlNetUnit] = None,
        adetailer_parameters: Optional[ADetailerArgs] = None,
        override_settings: Optional[OverRideSettings] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> List[str]:
        """
        Generate image(s) from the given input image using diffusion-based algorithms and refinement techniques.

        Args:
            diffusion_parameters (DiffusionParser): An instance of DiffusionParser class containing diffusion parameters.
            refiner_parameters (Optional[RefinerParser]): An optional instance of RefinerParser class containing refiner parameters.
            controlnet_parameters (Optional[ControlNetUnit]): An optional instance of ControlNetUnit class containing controlnet parameters.
            adetailer_parameters (Optional[ADetailerArgs]): An optional instance of ADetailerArgs class containing adetailer parameters.
            image_path (Optional[str]): An optional path to the input image file.
            image_base64 (Optional[str]): An optional base64 encoded string representation of the input image.
            override_settings (Optional[OverRideSettings]): An optional instance of OverRideSettings class containing override settings.

        Returns:
            List[str]: A list of file paths of the generated images.
        """
        # Create the payload dictionary to be sent in the request
        self.img2img_params.payload_init()

        if image_path:
            # Convert the input image to base64 and add it to the payload
            png_payload: Dict = {INIT_IMAGES_KEY: [img_to_base64(image_path)]}
        elif image_base64:
            png_payload: Dict = {INIT_IMAGES_KEY: [image_base64]}
        else:
            raise ValueError("one of image_path and image_base64 must be specified!")
        self.img2img_params.add_payload(
            merge_payloads(
                png_payload,
                diffusion_parameters.dict(),
                override_settings.dict() if override_settings else None,
                refiner_parameters.dict() if refiner_parameters else None,
            )
        )

        # Add the alwayson scripts to the payload
        self.img2img_params.add_payload(
            {
                ALWAYSON_SCRIPTS_KEY: merge_payloads(
                    adetailer_parameters.make_pyload() if adetailer_parameters else None,
                    make_cn_payload([controlnet_parameters]) if controlnet_parameters else None,
                )
            }
        )

        images_paths = await self._make_image_gen_request(self.img2img_params.current, API_IMG2IMG)

        self.img2img_params.store()

        return images_paths

    async def img2img_history(
        self,
    ) -> List[str]:
        return await self._make_image_gen_request(self.img2img_params.history[-1], API_IMG2IMG)

    async def txt2img_history(self) -> List[str]:
        return await self._make_image_gen_request(self.txt2img_params.history[-1], API_TXT2IMG)

    async def _make_image_gen_request(self, payload: Dict, image_gen_api) -> List[str]:
        """
        Makes a request to the image generation API with the given payload and saves the generated images to the output directory.

        Args:
            payload (Dict): The payload to be sent in the request.
            image_gen_api: The API endpoint for image generation.

        Returns:
            List[str]: The list of file paths for the saved images.
        """

        # Send a POST request to the API with the payload and get the response
        async with aiohttp.ClientSession() as session:
            response_payload: Dict = await (await session.post(f"{self.host_url}/{image_gen_api}", json=payload)).json()

        # Extract the generated images from the response payload
        img_base64: List[str] = extract_png_from_payload(response_payload)

        # Save the generated images to the output directory and return the list of file paths
        return save_base64_img_with_hash(img_base64_list=img_base64, output_dir=self.output_dir, host_url=self.host_url)

    async def _make_query_request(self, query_api: str, payload: Dict = None) -> Any:
        """
        Makes a query request to the specified query API.

        Args:
            payload ():
            query_api (str): The query API to make the request to.

        Returns:
            Any: The response payload from the query request.
        """
        full_url = f"{self.host_url}/{query_api}"
        async with aiohttp.ClientSession() as session:
            if payload:
                response_payload: Dict = await (await session.post(full_url, json=payload)).json()
            else:
                response_payload: Dict = await (await session.get(full_url)).json()
        return response_payload

    async def txt2img_favorite(self, index: Optional[int] = None) -> List[str]:
        return await self._make_image_gen_request(
            self.txt2img_params.favorite[index] if index else choice(self.txt2img_params.favorite), API_TXT2IMG
        )

    async def img2img_favorite(self, index: Optional[int] = None) -> List[str]:
        return await self._make_image_gen_request(
            self.img2img_params.favorite[index] if index else choice(self.img2img_params.favorite), API_IMG2IMG
        )

    async def fetch_sd_models(self) -> List[Dict]:
        models_detail_list: List[Dict] = await self._make_query_request(API_MODELS)
        self.available_sd_models.clear()
        self.available_sd_models.extend(map(lambda x: x["title"], models_detail_list))
        return models_detail_list

    async def fetch_lora_models(self) -> List[Dict]:
        models_detail_list: List[Dict] = await self._make_query_request(API_LORAS)
        self.available_lora_models.clear()
        self.available_lora_models.extend(map(lambda x: x["name"], models_detail_list))
        return models_detail_list

    async def fetch_upscalers(self) -> List[Dict]:
        models_detail_list: List[Dict] = await self._make_query_request(API_GET_UPSCALERS)
        self.available_upscalers.clear()
        self.available_upscalers.extend(map(lambda x: x["name"], models_detail_list))
        return models_detail_list

    async def interrogate_image(self, parser: InterrogateParser) -> OrderedDict[str, float]:
        return deepdanbooru_to_obj((await self._make_query_request(API_INTERROGATE, payload=parser.dict()))["caption"])


def deepdanbooru_to_obj(string: str) -> OrderedDict[str, float]:
    """
    Convert a string representation of a deepbooru object to a tuple of tuples.

    Args:
        string (str): The string representation of the deepbooru object.

    Returns:
        Tuple[Tuple[str, float], ...]: A tuple of tuples containing the parsed values.

    Example:
        >>> from collections import OrderedDict
        >>> deepdanbooru_to_obj("(tag1:0.6),(tag2:0.3),(tag3:0.8)")
        OderderedDict([('tag3', 0.8), ('tag1', 0.6), ('tag2', 0.3)])
    """

    split = map(lambda item: (item.strip()[1:-1]).split(":"), string.split(","))
    filtered = filter(lambda item: len(item) == 2, split)
    ref = map(lambda item: (item[0], float(item[1])), filtered)

    # 创建一个空的有序字典
    ordered_dict: OrderedDict = OrderedDict()
    for tag in sorted(ref, key=lambda x: x[1], reverse=True):
        ordered_dict[tag[0]] = tag[1]
    return ordered_dict
