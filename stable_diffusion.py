import asyncio
import copy
import json
import pathlib
import re
import warnings
from collections import OrderedDict
from random import choice
from typing import List, Dict, Optional, Any, Tuple

from PIL import PngImagePlugin, Image
from aiohttp import ClientSession, ClientConnectorError
from colorama import Fore
from pydantic import BaseModel, Field, validator

from modules.shared import img_to_base64, base64_to_img, rename_image_with_hash
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
    API_PNG_INFO,
    IMAGE_KEY,
    PNG_INFO_KEY,
    API_INTERRUPT,
    API_SAMPLERS,
)
from .controlnet import ControlNetUnit, make_cn_payload
from .parser import (
    DiffusionParser,
    HiResParser,
    OverRideSettings,
    InterrogateParser,
    RefinerParser,
)
from .utils import extract_png_from_payload


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

    class Config:
        arbitrary_types_allowed = True

    host_url: str
    cache_dir: str
    output_dir: str
    img2img_params: Optional[PersistentManager]
    txt2img_params: Optional[PersistentManager]
    available_sd_models: List[str] = Field(default_factory=list, const=True)
    available_lora_models: List[str] = Field(default_factory=list, const=True)
    available_upscalers: List[str] = Field(default_factory=list, const=True)
    available_samplers: List[str] = Field(default_factory=list, const=True)

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
        session: Optional[ClientSession] = None,
    ) -> List[str]:
        """
        Converts text to images.

        Args:
            diffusion_parameters (DiffusionParser, optional): The parameters for text diffusion.
                Defaults to DiffusionParser().
            hires_parameters (HiResParser, optional): The parameters for generating high-resolution images.
                Defaults to None.
            refiner_parameters (RefinerParser, optional): The parameters for image refinement.
                Defaults to None.
            controlnet_parameters (ControlNetUnit, optional): The parameters for control net.
                Defaults to None.
            adetailer_parameters (ADetailerArgs, optional): The parameters for adetailer.
                Defaults to None.
            override_settings (OverRideSettings, optional): The override settings for the parameters.
                Defaults to None.
            session (ClientSession, optional): The client session.
                Defaults to None.

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

        images_paths = await self._make_image_gen_request(self.txt2img_params.current, API_TXT2IMG, session)

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
        session: Optional[ClientSession] = None,
    ) -> List[str]:
        """
        Asynchronously converts an image to another image using various parameters.

        Args:
            diffusion_parameters (DiffusionParser): The parameters for the diffusion process.
            refiner_parameters (Optional[RefinerParser]): The parameters for the refiner process.
            controlnet_parameters (Optional[ControlNetUnit]): The parameters for the controlnet process.
            adetailer_parameters (Optional[ADetailerArgs]): The parameters for the adetailer process.
            override_settings (Optional[OverRideSettings]): The override settings for the process.
            image_path (Optional[str]): The path of the input image.
            image_base64 (Optional[str]): The base64 representation of the input image.
            session (Optional[ClientSession]): The client session for making the request.

        Returns:
            List[str]: A list of paths to the converted images.
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

        images_paths = await self._make_image_gen_request(self.img2img_params.current, API_IMG2IMG, session)

        self.img2img_params.store()

        return images_paths

    async def img2img_history(
        self,
    ) -> List[str]:
        return await self._make_image_gen_request(self.img2img_params.history[-1], API_IMG2IMG)

    async def txt2img_history(self) -> List[str]:
        return await self._make_image_gen_request(self.txt2img_params.history[-1], API_TXT2IMG)

    async def _make_image_gen_request(
        self,
        payload: Dict,
        image_gen_api: str,
        session: Optional[ClientSession] = None,
    ) -> List[str]:
        """
        Makes a request to the image generation API with the given payload and saves the generated images to the output directory.

        Args:
            payload (Dict): The payload to be sent in the request.
            image_gen_api: The API endpoint for image generation.

        Returns:
            List[str]: The list of file paths for the saved images.
        """
        is_local_session = session is None
        session = session or ClientSession(base_url=self.host_url)
        # Send a POST request to the API with the payload and get the response
        async with session.post(image_gen_api, json=payload) as response:
            response_payload: Dict = await response.json()
            # Extract the generated images from the response payload
            img_base64_chunks: List[str] = extract_png_from_payload(response_payload)
            png_infos: Tuple[str] = await asyncio.gather(
                *[
                    self.get_png_info(image_base64=img_base64_chunk, session=session)
                    for img_base64_chunk in img_base64_chunks
                ]
            )
        if is_local_session:
            await session.close()

        # Save the generated images to the output directory and return the list of file paths
        return [
            assemble_image(image_base64=img_base64_chunk, png_info=png_info, output_dir=self.output_dir)
            for img_base64_chunk, png_info in zip(img_base64_chunks, png_infos)
        ]

    async def _make_query_request(
        self,
        query_api: str,
        payload: Dict = None,
        method: Optional[str] = None,
        session: Optional[ClientSession] = None,
    ) -> Any:
        """
        Makes a query request to the specified query API.

        Args:
            payload ():
            query_api (str): The query API to make the request to.

        Returns:
            Any: The response payload from the query request.
        """
        print(f"{Fore.LIGHTBLUE_EX}Making query ==> {query_api}{Fore.RESET}")
        is_local_session = session is None
        session = session or ClientSession(base_url=self.host_url)

        method = method or ("POST" if payload else "GET")
        async with session.request(method, query_api, json=payload) as response:
            ret = await response.json()
        if is_local_session:
            await session.close()

        return ret

    async def txt2img_favorite(self, index: Optional[int] = None) -> List[str]:
        return await self._make_image_gen_request(
            self.txt2img_params.favorite[index] if index else choice(self.txt2img_params.favorite), API_TXT2IMG
        )

    async def img2img_favorite(self, index: Optional[int] = None) -> List[str]:
        return await self._make_image_gen_request(
            self.img2img_params.favorite[index] if index else choice(self.img2img_params.favorite), API_IMG2IMG
        )

    async def interrupt(self, session: Optional[ClientSession] = None) -> None:
        """
        Interrupts the current process by making a POST request to the interrupt API.

        Args:
            session (Optional[ClientSession]): The session to use for the request. If not provided,
                a new session will be created with the default base URL.

        Returns:
            None

        Raises:
            None
        """
        is_local_session = session is None
        session = session or ClientSession(base_url=self.host_url)
        await self._make_query_request(API_INTERRUPT, method="POST", session=session)
        if is_local_session:
            await session.close()

    async def _fetch_and_store_models(
        self, api_path: str, container: List[str], key_to_extract: str, session: Optional[ClientSession]
    ) -> List[Dict]:
        try:
            models_detail_list: List[Dict] = await self._make_query_request(api_path, session=session)
        except ClientConnectorError as e:
            warnings.warn(f"Can't fetch models from {api_path}, {e}")
            return []

        container.clear()
        container.extend(map(lambda x: x[key_to_extract], models_detail_list))
        container.sort()
        return models_detail_list

    async def fetch_sd_models(self, session: Optional[ClientSession] = None) -> List[Dict]:
        return await self._fetch_and_store_models(API_MODELS, self.available_sd_models, "title", session)

    async def fetch_lora_models(self, session: Optional[ClientSession] = None) -> List[Dict]:
        return await self._fetch_and_store_models(API_LORAS, self.available_lora_models, "name", session)

    async def fetch_upscalers(self, session: Optional[ClientSession] = None) -> List[Dict]:
        return await self._fetch_and_store_models(API_GET_UPSCALERS, self.available_upscalers, "name", session)

    async def fetch_sampler(self, session: Optional[ClientSession] = None) -> List[Dict]:
        return await self._fetch_and_store_models(API_SAMPLERS, self.available_samplers, "name", session)

    async def interrogate_image(self, parser: InterrogateParser) -> OrderedDict[str, float]:
        """
        Interrogates an image using the given parser and returns the result as an ordered dictionary.

        Args:
            parser (InterrogateParser): The parser object containing the image information.

        Returns:
            OrderedDict[str, float]: The result of the interrogation as a dictionary with caption as the key and confidence score as the value.
        """
        return deepdanbooru_to_obj((await self._make_query_request(API_INTERROGATE, payload=parser.dict()))["caption"])

    async def get_png_info(
        self,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        session: Optional[ClientSession] = None,
    ) -> str:
        """
        Fetches information about a PNG image.

        Args:
            image_path (str): The path to the PNG image file.
            image_base64 (Optional[str], optional): The base64-encoded PNG image. Defaults to None.
            session (Optional[ClientSession], optional): The aiohttp ClientSession to use for making the request.
                Defaults to None.

        Returns:
            str: The information about the PNG image.

        Raises:
            None

        """
        image = [image_path, image_base64]
        if not any(image):
            raise ValueError("One of image_path or image_base64 should be provided.")
        if all(image):
            raise ValueError("Only one of image_path or image_base64 should be provided.")
        image_base64: str = image_base64 or img_to_base64(image_path)
        if session:
            async with session.post(url=API_PNG_INFO, json={IMAGE_KEY: image_base64}) as response:
                return (await response.json()).get(PNG_INFO_KEY)
        else:
            async with ClientSession(base_url=self.host_url) as selected_session:
                async with selected_session.post(url=API_PNG_INFO, json={IMAGE_KEY: image_base64}) as response:
                    return (await response.json()).get(PNG_INFO_KEY)


def inject_png_info(image_path: str, req_png_info: str) -> None:
    """
    Injects PNG metadata into an image file.

    Args:
        image_path (str): The path to the image file.
        req_png_info (str): The PNG metadata to be injected.

    Returns:
        None: This function does not return anything.
    """
    image = Image.open(image_path)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("parameters", req_png_info)
    image.save(image_path, pnginfo=png_info)


def assemble_image(image_base64: str, png_info: str, output_dir: str, max_name_length: int = 50) -> str:
    """
    Assembles an image from a base64 string, adds PNG information, saves it to the specified output directory,
        and returns the path of the saved image.

    Args:
        image_base64 (str): The base64 string representing the image.
        png_info (str): The PNG information to be added to the image.
        output_dir (str): The directory where the image will be saved.
        max_name_length (int, optional): The maximum length of the file name. Defaults to 50.

    Returns:
        str: The path of the saved image.
    """
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_name = re.sub(r'[\\/:"*?<>|]', "", png_info) or "null"
    if len(file_name) > max_name_length:
        file_name = file_name[:max_name_length]

    output_path = f"{output_dir}/{file_name}.png"
    base64_to_img(image_base64, output_path)
    inject_png_info(output_path, png_info)

    return rename_image_with_hash(output_path)


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
