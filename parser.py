import random
from typing import List, Tuple

from pydantic import BaseModel, Field

__DEFAULT_NEGATIVE_PROMPT__ = (
    "loathed,low resolution,porn,NSFW,strange shaped finger,cropped,panties visible,"
    "pregnant,ugly,vore,duplicate,extra fingers,fused fingers,too many fingers,mutated hands,"
    "poorly drawn face,mutation,bad anatomy,blurry,malformed limbs,disfigured,extra limbs,"
    "missing arms,missing legs,extra arms,deformed legs, bad anatomy, bad hands, "
    "text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, "
    "low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, "
    "bad feet,,poorly drawn asymmetric eyes,cloned face,limbs,mutilated,deformed,malformed,"
    "multiple breasts,missing fingers,poorly drawn,poorly drawn hands,extra legs,"
    "mutated hands and fingers,bad anatomy disfigured malformed mutated,worst quality,"
    "too many fingers,malformed hands,Missing limbs,long neck,blurry,missing arms,three arms,"
    "long body,more than 2 thighs,more than 2 nipples,missing legs,mutated hands and fingers ,"
    "low quality,jpeg artifacts,signature,extra digit,fewer digits,lowres,bad anatomy,extra limbs,"
)

__DEFAULT_POSITIVE_PROMPT__ = (
    "modern art,student uniform,white shirt,short blue skirt,white tights,joshi,JK,1girl:1.2,solo,upper body,"
    "shy,extremely cute,lovely,fish eye,outside,on street"
    "beautiful,expressionless,cool girl,medium breasts,water color,oil,see through"
    "thighs,thin torso,masterpiece,wonderful art,high resolution,hair ornament,strips,body curve,hair,SFW:1.3,"
)
__R_GEN__ = random.SystemRandom()


def set_default_pos_prompt(new_prompt: str) -> bool:
    """
    Sets the default positive prompt.

    Args:
        new_prompt (str): The new prompt to be set.

    Returns:
        bool: True if the prompt was set successfully, False otherwise.
    """
    global __DEFAULT_POSITIVE_PROMPT__
    __DEFAULT_POSITIVE_PROMPT__ = new_prompt
    return True


def get_default_pos_prompt() -> str:
    """
    Returns the default positive prompt as a string.
    """
    global __DEFAULT_POSITIVE_PROMPT__
    return __DEFAULT_POSITIVE_PROMPT__


def set_default_neg_prompt(new_prompt: str) -> bool:
    """
    Set the default negative prompt.

    Args:
        new_prompt (str): The new negative prompt to set.

    Returns:
        bool: True if the default negative prompt was successfully set.
    """
    global __DEFAULT_NEGATIVE_PROMPT__
    __DEFAULT_NEGATIVE_PROMPT__ = new_prompt
    return True


def get_default_neg_prompt() -> str:
    """
    Returns the default negative prompt.

    :return: The default negative prompt.
    :rtype: str
    """
    global __DEFAULT_NEGATIVE_PROMPT__
    return __DEFAULT_NEGATIVE_PROMPT__


def get_seed() -> int:
    """
    Generate a random 32-bit integer as a seed.

    Returns:
        int: A 32-bit integer generated as a seed.
    """
    return __R_GEN__.getrandbits(32)


__WIDE_SHOT__: Tuple[int, int] = (768, 512)
__PORTRAIT_SHOT__: Tuple[int, int] = (512, 768)
__SQUARE_SHOT__: Tuple[int, int] = (512, 512)
__ENFORCED_SIZE_TEMPLATE__: Tuple[int, int] = __PORTRAIT_SHOT__


__SHOT_SIZE_TABLE__ = {
    "wide": __WIDE_SHOT__,
    "portrait": __PORTRAIT_SHOT__,
    "square": __SQUARE_SHOT__,
}


def set_shot_size(label: str) -> bool:
    """
    Set the shot size based on the given label.

    Args:
        label (str): The label to determine the shot size.

    Returns:
        bool: True if the label is found in the shot size table, False otherwise.
    """
    global __ENFORCED_SIZE_TEMPLATE__, __SHOT_SIZE_TABLE__

    __ENFORCED_SIZE_TEMPLATE__ = __SHOT_SIZE_TABLE__.get(label, __PORTRAIT_SHOT__)
    return label in __SHOT_SIZE_TABLE__


def get_shot_size() -> Tuple[int, int]:
    global __ENFORCED_SIZE_TEMPLATE__
    return __ENFORCED_SIZE_TEMPLATE__


def get_shot_width() -> int:
    global __ENFORCED_SIZE_TEMPLATE__
    return __ENFORCED_SIZE_TEMPLATE__[0]


def get_shot_height() -> int:
    global __ENFORCED_SIZE_TEMPLATE__
    return __ENFORCED_SIZE_TEMPLATE__[1]


class DiffusionParser(BaseModel):
    """
    use to parse config_registry
    """

    prompt: str = Field(default_factory=get_default_pos_prompt)
    negative_prompt: str = Field(default_factory=get_default_neg_prompt)
    styles: List[str] = Field(default_factory=list)
    seed: int = -1
    sampler_name: str = "UniPC"
    steps: int = 18
    cfg_scale: float = 6.9
    width: int = Field(default_factory=get_shot_width)
    height: int = Field(default_factory=get_shot_height)


class HiResParser(BaseModel):
    """
    use to parse hires config_registry
    """

    enable_hr: bool = False
    denoising_strength: float = 0.69
    hr_scale: float = 1.3
    hr_upscaler: str = "Latent"
    # hr_checkpoint_name: string
    # hr_sampler_name: string
    # hr_prompt:
    # hr_negative_prompt:
