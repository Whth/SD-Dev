import base64
import io
import os
import re
from random import shuffle
from typing import List, Dict, Callable, Any, Tuple

from PIL import Image, PngImagePlugin
from aiohttp import ClientSession
from colorama import Fore
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


async def save_base64_img_with_hash(
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
    async with ClientSession(base_url=host_url) as sd_session:
        for img_base64 in img_base64_list:
            # Decode the base64-encoded image

            response = await sd_session.post(url=API_PNG_INFO, json={IMAGE_KEY: img_base64})
            req_png_info = (await response.json()).get(PNG_INFO_KEY)

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
    string: str,
    pos_keyword: str = "+",
    neg_keyword: str = "-",
    batch_count_keyword: str = "pP",
    raise_settings: Tuple[bool, bool, bool] = (False, False, False),
) -> Tuple[str, str, int]:
    """
    Extracts prompts from a given string.

    Args:
        string (str): The input string to extract prompts from.
        pos_keyword (str, optional): The positive prompt keyword. Defaults to "+".
        neg_keyword (str, optional): The negative prompt keyword. Defaults to "-".
        batch_count_keyword (str, optional): The batch count keyword. Defaults to "pP".
        raise_settings (Tuple[bool, bool, bool], optional): A tuple of booleans indicating whether to raise an exception if a prompt is not found. Defaults to (False, False, False).

    Returns:
        Tuple[str, str, int]: A tuple containing the positive prompt, negative prompt, and batch count extracted from the string.
    """

    # Match positive prompt
    pos_matched = re.compile(rf".*?([{pos_keyword}])(.*?)\1").match(string=string)

    # Match negative prompt
    neg_matched = re.compile(rf".*?([{neg_keyword}])(.*?)\1").match(string=string)

    # Match batch count
    count_matched = re.compile(rf".*?(\d+)[{batch_count_keyword}]").match(string=string)

    # Check if positive prompt is required and not found
    if raise_settings[0] and not pos_matched:
        raise ValueError(f"Positive prompt not found in string: {string}")

    # Check if negative prompt is required and not found
    if raise_settings[1] and not neg_matched:
        raise ValueError(f"Negative prompt not found in string: {string}")

    # Check if batch count is required and not found
    if raise_settings[2] and not count_matched:
        raise ValueError(f"Batch count not found in string: {string}")

    # Return the extracted prompts and batch count
    return (
        (pos_matched.group(2) if pos_matched else ""),
        (neg_matched.group(2) if neg_matched else ""),
        int(count_matched.group(1)) if count_matched else 1,
    )


class PromptProcessorRegistry(object):
    def __init__(self):
        self._registry_list: List[Tuple[Callable[[], Any], Callable[[str, str], Tuple[str, str]]]] = []
        self._process_name: List[str] = []

    def process(self, pos_prompt: str, neg_prompt: str) -> Tuple[str, str]:
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


def split_list(input_list: List, sublist_size: int, strip_remains: bool = False) -> List[List]:
    """
    Splits a given list into sublist of a specified size.

    Args:
        input_list (List): The list to be split.
        sublist_size (int): The size of each sublist.
        strip_remains (bool, optional): Determines whether to strip any remaining elements. Defaults to False.

    Returns:
        List[List]: The list of sublist.
    """
    # Create sublist of the specified size
    result = [input_list[i : i + sublist_size] for i in range(0, len(input_list), sublist_size)]

    # Check if remains should be stripped
    if strip_remains:
        result = result[:-1]

    return result


def make_lora_replace_process_engine(lora_list: List[str], identifier: str = "lr") -> Callable:
    pat = re.compile(rf"({identifier}:(\d+)(?::(\d+\.\d+))?)")

    def lora_wrapper(lora: str, strength: str) -> str:
        if not strength:
            strength = 1.0
        return f"<lora:{lora}:{strength}>"

    def process_engine(prompt: str) -> str:
        matched = pat.findall(prompt)
        if matched:
            for m in matched:
                prompt = prompt.replace(m[0], lora_wrapper(lora_list[int(m[1])], m[2]))
        return prompt

    return process_engine
