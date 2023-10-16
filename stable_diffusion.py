import base64
import copy
import io
import json
import os.path
import pathlib
from random import choice
from typing import NamedTuple, List, Dict, Optional

import aiohttp
import requests
from PIL import Image, PngImagePlugin
from pydantic import BaseModel, Field, validator
from slugify import slugify

from modules.file_manager import rename_image_with_hash, img_to_base64
from .api import (
    API_PNG_INFO,
    API_TXT2IMG,
    INIT_IMAGES_KEY,
    IMAGE_KEY,
    IMAGES_KEY,
    PNG_INFO_KEY,
    ALWAYSON_SCRIPTS_KEY,
)
from .controlnet import ControlNetUnit, make_cn_payload

DEFAULT_NEGATIVE_PROMPT = "loathed,low resolution,porn,NSFW,strange shaped finger,cropped,panties visible"

DEFAULT_POSITIVE_PROMPT = (
    "modern art,student uniform,white shirt,short blue skirt,white tights,joshi,JK,1girl:1.2,solo,upper body,"
    "shy,extremely cute,lovely,"
    "beautiful,expressionless,cool girl,medium breasts,"
    "thighs,thin torso,masterpiece,wonderful art,high resolution,hair ornament,strips,body curve,hair,SFW:1.3,"
)


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


class DiffusionParser(NamedTuple):
    """
    use to parse config
    """

    prompt: str = DEFAULT_POSITIVE_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    styles: List[str] = []
    seed: int = -1
    sampler_name: str = "UniPC"
    steps: int = 18
    cfg_scale: float = 6.9
    width: int = 512
    height: int = 768


class HiResParser(NamedTuple):
    """
    use to parse hires config
    """

    enable_hr: bool = False
    denoising_strength: float = 0.69
    hr_scale: float = 1.3
    hr_upscaler: str = "Latent"
    # hr_checkpoint_name: string
    # hr_sampler_name: string
    # hr_prompt:
    # hr_negative_prompt:


I2I_HISTORY_SAVE_PATH = "i2i_history.json"
T2I_HISTORY_SAVE_PATH = "t2i_history.json"


class StableDiffusionApp(object):
    """
    class that implements the basic diffusion api
    """

    def __init__(self, host_url: str, cache_dir: str):
        self._host_url: str = host_url
        self._cache_dir: str = cache_dir
        self._img2img_params: PersistentManager = PersistentManager(
            save_path=f"{self._cache_dir}/{I2I_HISTORY_SAVE_PATH}"
        )
        self._txt2img_params: PersistentManager = PersistentManager(
            save_path=f"{self._cache_dir}/{T2I_HISTORY_SAVE_PATH}"
        )

    @property
    def img2img_params(self) -> PersistentManager:
        return self._img2img_params

    @property
    def txt2img_params(self) -> PersistentManager:
        return self._txt2img_params

    def add_favorite_i(self) -> bool:
        self._img2img_params.add_favorite()
        return True

    def add_favorite_t(self) -> bool:
        self._txt2img_params.add_favorite()
        return True

    async def txt2img(
        self,
        output_dir: str,
        diffusion_parameters: DiffusionParser = DiffusionParser(),
        HiRes_parameters: HiResParser = HiResParser(),
        controlnet_parameters: Optional[ControlNetUnit] = None,
    ) -> List[str]:
        """
        Generates images from text files and saves them to the specified output directory.

        Args:
            output_dir (str): The path to the directory where the generated images will be saved.
            diffusion_parameters (DiffusionParser, optional): An instance of the DiffusionParser class
                that contains the parameters for the diffusion process.
                Defaults to DiffusionParser().
            HiRes_parameters (HiResParser, optional): An instance of the HiResParser class
                that contains the parameters for the HiRes process.
                Defaults to HiResParser().
            controlnet_parameters (ControlNetUnit, optional): An instance of the ControlNetUnit class
                that contains the parameters for the controlnet process.
                Defaults to None.

        Returns:
            List[str]: A list of paths to the generated images.
        """

        self._txt2img_params.payload_init()
        alwayson_scripts: Dict = {ALWAYSON_SCRIPTS_KEY: {}}
        self._txt2img_params.add_payload(diffusion_parameters._asdict())
        self._txt2img_params.add_payload(HiRes_parameters._asdict())

        if controlnet_parameters:
            alwayson_scripts[ALWAYSON_SCRIPTS_KEY].update(make_cn_payload([controlnet_parameters]))

        self._txt2img_params.add_payload(alwayson_scripts)
        images_paths = await self._make_request(output_dir, self._txt2img_params.current)

        self._txt2img_params.store()

        return images_paths

    async def img2img(
        self,
        output_dir: str,
        diffusion_parameters: DiffusionParser = DiffusionParser(),
        controlnet_parameters: Optional[ControlNetUnit] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> List[str]:
        """
        Converts an image to another image using the specified diffusion parameters and
        controlnet parameters (optional).
        Saves the generated images to the specified output directory.

        Args:
            image_base64 ():
            image_path: The path of the input image file.
            output_dir: The directory where the generated images will be saved.
            diffusion_parameters: An instance of DiffusionParser class containing the diffusion parameters.
            controlnet_parameters: An optional instance of ControlNetUnit class containing the controlnet parameters.

        Returns:
            A list of file paths of the saved generated images.
        """

        # Create the payload dictionary to be sent in the request
        self._img2img_params.payload_init()

        # Create a dictionary to store the alwayson scripts
        alwayson_scripts: Dict = {ALWAYSON_SCRIPTS_KEY: {}}

        # Add the diffusion parameters to the payload
        self._img2img_params.add_payload(diffusion_parameters._asdict())

        if image_path:
            # Convert the input image to base64 and add it to the payload
            png_payload: Dict = {INIT_IMAGES_KEY: [img_to_base64(image_path)]}
        elif image_base64:
            png_payload: Dict = {INIT_IMAGES_KEY: [image_base64]}
        else:
            raise ValueError("one of image_path and image_base64 must be specified!")
        self._img2img_params.add_payload(png_payload)

        # If controlnet parameters are provided, update the alwayson scripts with them
        if controlnet_parameters:
            alwayson_scripts[ALWAYSON_SCRIPTS_KEY].update(make_cn_payload([controlnet_parameters]))

        # Add the alwayson scripts to the payload
        self._img2img_params.add_payload(alwayson_scripts)

        images_paths = await self._make_request(output_dir, self._img2img_params.current)

        self._img2img_params.store()

        return images_paths

    async def img2img_history(
        self,
        output_dir: str,
    ) -> List[str]:
        return await self._make_request(output_dir, self._img2img_params.history[-1])

    async def txt2img_history(self, output_dir: str) -> List[str]:
        return await self._make_request(output_dir, self._txt2img_params.history[-1])

    async def _make_request(self, output_dir, payload: Dict) -> List[str]:
        """
        Makes a request to the API and saves the generated images to the output directory.

        Args:
            output_dir: The directory where the generated images will be saved.
            payload: The payload to be sent in the request.

        Returns:
            A list of paths to the saved images.
        """
        # Send a POST request to the API with the payload and get the response
        async with aiohttp.ClientSession() as session:
            response_payload: Dict = await (await session.post(f"{self._host_url}/{API_TXT2IMG}", json=payload)).json()
        # Extract the generated images from the response payload
        img_base64: List[str] = extract_png_from_payload(response_payload)
        # Save the generated images to the output directory and return the list of file paths
        return save_base64_img_with_hash(img_base64_list=img_base64, output_dir=output_dir, host_url=self._host_url)

    async def txt2img_favorite(self, output_dir: str, index: Optional[int] = None) -> List[str]:
        return await self._make_request(
            output_dir, self._txt2img_params.favorite[index] if index else choice(self._txt2img_params.favorite)
        )

    async def img2img_favorite(self, output_dir: str, index: Optional[int] = None) -> List[str]:
        return await self._make_request(
            output_dir, self._img2img_params.favorite[index] if index else choice(self._img2img_params.favorite)
        )


def extract_png_from_payload(payload: Dict) -> List[str]:
    """
    Should extract a list of png encoded of base64

    Args:
        payload (Dict): the response payload

    Returns:

    """

    if IMAGES_KEY not in payload:
        raise KeyError(f"{IMAGES_KEY} not found in payload")
    img_base64 = payload.get(IMAGES_KEY)
    return img_base64


def save_base64_img_with_hash(
    img_base64_list: List[str], output_dir: str, host_url: str, max_file_name_length: int = 34
) -> List[str]:
    """
    Process a list of base64-encoded images and save them as PNG files in the specified output directory.

    :param img_base64_list: A list of base64-encoded images.
    :param output_dir: The directory where the output images will be saved.
    :param host_url: The URL of the host where the API is running.
    :param max_file_name_length: The maximum length of the file name.
    :return: A list of paths to the saved images.
    """

    output_img_paths: List[str] = []

    for img_base64 in img_base64_list:
        # Decode the base64-encoded image

        # Make a POST request to get PNG info
        response = requests.post(url=f"{host_url}/{API_PNG_INFO}", json={IMAGE_KEY: img_base64})

        req_png_info = response.json().get(PNG_INFO_KEY)

        # Create a label for the saved image
        label = slugify(
            req_png_info[:max_file_name_length] if len(req_png_info) > max_file_name_length else req_png_info
        )

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Save the image with the PNG info
        saved_path = f"{output_dir}/{label}.png"
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("parameters", req_png_info)
        image = Image.open(io.BytesIO(base64.b64decode(img_base64)))
        image.save(saved_path, pnginfo=png_info)

        # Rename the saved image with a hash
        saved_path_with_hash = rename_image_with_hash(saved_path)

        # Add the path to the list of output image paths
        output_img_paths.append(saved_path_with_hash)

    return output_img_paths


def get_image_ratio(image_path):
    """
    获取图片长宽比
    :param image_path: 图片路径
    :return: 图片长宽比
    """

    img = Image.open(image_path)
    width, height = img.size
    return width / height
