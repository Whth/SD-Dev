import base64
import io
import os.path
import os.path
import re
from random import shuffle
from typing import List, Dict, Tuple, Callable, Any

import requests
from PIL import Image
from PIL import PngImagePlugin
from slugify import slugify

from modules.file_manager import rename_image_with_hash
from .api import (
    API_PNG_INFO,
    IMAGE_KEY,
    IMAGES_KEY,
    PNG_INFO_KEY,
)


def get_image_ratio(image_path) -> float:
    """
    获取图片长宽比
    :param image_path: 图片路径
    :return: 图片长宽比
    """

    img = Image.open(image_path)
    width, height = img.size
    return width / height


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
    Saves a list of base64-encoded images as PNG files with hash-based filenames.

    Args:
        img_base64_list (List[str]): A list of base64-encoded images.
        output_dir (str): The directory where the output images will be saved.
        host_url (str): The URL of the host server.
        max_file_name_length (int, optional): The maximum length of the file name before truncation. Defaults to 34.

    Returns:
        List[str]: A list of paths to the saved PNG images with hash-based filenames.
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


def extract_prompts(
    message: str,
    specify_batch_count: bool = False,
    pos_keyword: List[str] = ("+",),
    neg_keyword: List[str] = ("-",),
    batch_count_keyword: List[str] = ("p", "P"),
) -> Tuple[List[str], List[str], int] | Tuple[List[str], List[str]]:
    """
    Extracts prompts from a given message.

    Args:
        message (str): The input message from which prompts are extracted.
        specify_batch_count (bool, optional): Specifies whether to specify the batch size. Defaults to False.
        pos_keyword (List[str], optional): The positive keywords used for prompt extraction. Defaults to ["+"].
        neg_keyword (List[str], optional): The negative keywords used for prompt extraction. Defaults to ["-"].
        batch_count_keyword (List[str], optional): The keywords used for batch size extraction. Defaults to ["p", "P"].

    Returns:
        Tuple[List[str], List[str], int] or Tuple[List[str], List[str]]: A tuple containing the lists of positive
            prompts and negative prompts extracted from the message.
            If `specify_batch_size` is True, it also returns the batch size as an integer.
    """
    if message == "":
        return [""], [""]
    pos_regx = "".join(pos_keyword)
    neg_regx = "".join(neg_keyword)
    pat = re.compile(rf"(?:([{pos_regx}])(.*?)\1)?(?:([{neg_regx}])(.*?)\3)?")
    matched_list = pat.findall(string=message)
    pos_prompt = list(map(lambda match: match[1], filter(lambda match: match[1], matched_list)))
    neg_prompt = list(map(lambda match: match[2], filter(lambda match: match[2], matched_list)))
    if specify_batch_count:
        batch_count_regx = "".join(batch_count_keyword)
        batch_count_pattern = re.compile(rf"(?:(\d+)[{batch_count_regx}])?")
        matched: List[str] = list(filter(bool, batch_count_pattern.findall(message)))
        if matched:
            return pos_prompt, neg_prompt, int(matched[0])

        return pos_prompt, neg_prompt, 1

    return pos_prompt, neg_prompt


class PromptProcessorRegistry(object):
    def __init__(self):
        self._registry_list: List[Tuple[Callable[[], Any], Callable[[str, str], Tuple[str, str]]]] = []
        self._process_name: List[str] = []

    def process(self, pos_prompt: str, neg_prompt: str) -> Tuple[str, str]:
        from colorama import Fore

        """
        Process the given positive and negative prompts using the registered processors.

        Args:
            pos_prompt (str): The positive prompt to be processed.
            neg_prompt (str): The negative prompt to be processed.

        Returns:
            Tuple[str, str]: A tuple containing the processed positive prompt and the processed negative prompt.
        """
        for processor, name in zip(self._registry_list, self._process_name):
            if processor[0]():
                pos_prompt, neg_prompt = processor[1](pos_prompt, neg_prompt)
                print(
                    f"\n{Fore.YELLOW}Executing {name}\n"
                    f"{self.prompt_string_constructor(pos_prompt=pos_prompt, neg_prompt=neg_prompt)}"
                )

        return pos_prompt, neg_prompt

    def register(
        self,
        judge: Callable[[], Any],
        process_name: str = None,
        processor: Callable[[str, str], Tuple[str, str]] = None,
        process_engine: Callable[[str], str] = None,
    ) -> None:
        """
        Register a judge and processor for the given process.

        Args:
            judge (Callable[[], Any]): A callable that takes no arguments and returns a value.
            process_name (str, optional): The name of the process. Defaults to None.
            processor (Callable[[str, str], Tuple[str, str]], optional): A callable that takes two strings as arguments and returns a tuple of two strings. Defaults to None.
            process_engine (Callable[[str], str], optional): A callable that takes a string as argument and returns a string. Defaults to None.

        Raises:
            ValueError: If neither a processor nor an engine is provided.

        Returns:
            None: This function does not return any value.
        """
        if not processor and not process_engine:
            raise ValueError("Either a processor or an engine must be provided.")
        processor = processor or self._make_processor_from_engine(process_engine)
        self._registry_list.append((judge, processor))
        if not process_name:
            process_name = f"Process-{len(self._registry_list)}"
        self._process_name.append(process_name)

    @staticmethod
    def _make_processor_from_engine(process_engine: Callable[[str], str]) -> Callable[[str, str], Tuple[str, str]]:
        def processor(pos_prompt: str, neg_prompt: str) -> Tuple[str, str]:
            return process_engine(pos_prompt), process_engine(neg_prompt)

        return processor

    @staticmethod
    def prompt_string_constructor(pos_prompt: str, neg_prompt: str) -> str:
        """
        Generate a formatted string containing positive and negative prompts.

        Args:
            pos_prompt (str): The positive prompt.
            neg_prompt (str): The negative prompt.

        Returns:
            str: A formatted string containing the positive and negative prompts.
        """
        from colorama import Fore

        return (
            f"{Fore.MAGENTA}___________________________________________\n"
            f"{Fore.GREEN}POSITIVE PROMPT:\n\t{pos_prompt}\n"
            f"{Fore.RED}NEGATIVE PROMPT:\n\t{neg_prompt}\n"
            f"{Fore.MAGENTA}___________________________________________\n{Fore.RESET}"
        )


def shuffle_prompt(prompt: str) -> str:
    temp = prompt.split(",")
    shuffle(temp)
    return ",".join(temp)
