import pathlib
import random
from functools import partial
from typing import List, Optional, Callable, Union, OrderedDict, Set

from aiohttp import ClientSession, ClientTimeout
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager
from graia.ariadne import Ariadne
from graia.ariadne.event.lifecycle import ApplicationLaunch
from graia.ariadne.event.message import GroupMessage, FriendMessage
from graia.ariadne.message.chain import MessageChain, Image
from graia.ariadne.message.parser.base import MatchRegex, ContainKeyword
from graia.ariadne.model import Group, Friend

from modules.shared import (
    make_stdout_seq_string,
    CmdBuilder,
    ExecutableNode,
    NameSpaceNode,
    download_file,
    get_pwd,
    img_to_base64,
    AbstractPlugin,
    make_regex_part_from_enum,
    assemble_cmd_regex_parts,
    EnumCMD,
    dict_to_markdown_table_complex,
)
from .adetailer import ADetailerArgs, ADetailerUnit, ModelType
from .controlnet import ControlNetUnit, Controlnet, ControlNetDetect
from .extractors import get_image_url, make_image_form_paths
from .lora_man import LoraManager
from .parser import (
    set_default_pos_prompt,
    set_default_neg_prompt,
    set_shot_size,
    get_default_neg_prompt,
    get_default_pos_prompt,
    Options,
    OverRideSettings,
    InterrogateParser,
    RefinerParser,
)
from .stable_diffusion import StableDiffusionApp, DiffusionParser, HiResParser
from .utils import (
    extract_prompts,
    PromptProcessorRegistry,
    shuffle_prompt,
    split_list,
    make_lora_replace_process_engine,
)

__all__ = ["StableDiffusionPlugin"]


class CMD(EnumCMD):
    stablediffusion = ["sd", "stbdf"]
    again = ["a", "ag", "rc"]
    img2img = ["i", "i2i"]
    txt2img = ["t", "t2i"]
    like = ["l", "lk"]
    size = ["s", "sz"]
    default = ["d", "dfa"]
    set = ["s", "st"]
    config = ["c", "cf", "cfg"]
    list = ["l", "ls"]
    models = ["m", "md"]
    modules = ["o", "mu"]
    interrogate = ["i", "inte"]
    normal = ["n", "s", "sd"]
    lora = ["l", "lr"]
    upscaler = ["u", "up", "ups"]
    sampler = ["sp", "samp"]
    positive = ["p", "pos"]
    negative = ["n", "neg"]

    test = ["t", "ts"]
    shot = ["s", "shts"]

    controlnet = ["cn", "ctln"]
    detect = ["d", "dtc"]
    fetch = ["f", "fch"]
    halt = ["h", "ht"]
    explain = ["x", "xp"]
    randlora = ["rlr"]

    yield_lora = ["y", "ylr"]
    remove_lora = ["r", "rml"]
    clean_lora = ["clr"]
    save_lora = ["slr"]
    what_lora = ["wlr"]
    weight = ["w", "wt"]


class StableDiffusionPlugin(AbstractPlugin):
    __TRANSLATE_PLUGIN_NAME: str = "BaiduTranslater"
    __TRANSLATE_METHOD_NAME: str = "translate"
    __TRANSLATE_METHOD_TYPE = Callable[[str, str, str], str]  # [tolang, query, fromlang] -> str

    TXT2IMG_DIRNAME = "txt2img"
    IMG2IMG_DIRNAME = "img2img"

    CONFIG_OUTPUT_DIR_PATH = "output_dir_path"
    CONFIG_IMG_TEMP_DIR_PATH = "temp"

    CONFIG_SD_HOST = "sd_host"

    CONFIG_POS_KEYWORD = "positive_keyword"
    CONFIG_NEG_KEYWORD = "negative_keyword"

    CONFIG_WILDCARD_DIR_PATH = "wildcard_dir_path"

    CONFIG_CONTROLNET_MODULE = "controlnet_module"
    CONFIG_CONTROLNET_MODEL = "controlnet_model"
    CONFIG_CLIP: str = "clip"
    CONFIG_STYLES = "styles"
    CONFIG_ENABLE_HR = "enable_hr"
    CONFIG_DENO_STRENGTH = "denoise_strength"
    CONFIG_HR_SCALE = "hr_scale"
    CONFIG_UPSCALER = "upscaler"
    CONFIG_SAMPLER: str = "sampler"
    CONFIG_CFG_SCALE: str = "cfg_scale"
    CONFIG_ENABLE_TRANSLATE = "enable_translate"
    CONFIG_ENABLE_CONTROLNET = "enable_controlnet"
    CONFIG_ENABLE_ADETAILER: str = "enable_adetailer"
    CONFIG_ENABLE_SHUFFLE_PROMPT = "enable_shuffle_prompt"
    CONFIG_ENABLE_DYNAMIC_PROMPT = "enable_dynamic_prompt"
    CONFIG_CURRENT_MODEL_ID = "crmodel"
    CONFIG_REFINER_MODEL_ID = "rfmodel"
    CONFIG_ENABLE_REFINER = "enable_refiner"
    CONFIG_SEND_BATCH_SIZE = "send_batch_size"
    CONFIG_UNIT_TIMEOUT = "utimeout"
    CONFIG_SWITCH_AT = "switchat"
    CONFIG_APPEND_LORAS = "append_loras"
    CONFIG_STEPS: str = "steps"
    DefaultConfig = {
        CONFIG_POS_KEYWORD: "+",
        CONFIG_NEG_KEYWORD: "-",
        CONFIG_OUTPUT_DIR_PATH: f"{get_pwd()}/output",
        CONFIG_IMG_TEMP_DIR_PATH: f"{get_pwd()}/temp",
        CONFIG_SD_HOST: "http://localhost:7860",
        # TODO: support multiple hosts
        CONFIG_WILDCARD_DIR_PATH: f"{get_pwd()}/asset/wildcard",
        CONFIG_CONTROLNET_MODULE: "openpose_full",
        CONFIG_CONTROLNET_MODEL: "control_v11p_sd15_openpose",
        CONFIG_CURRENT_MODEL_ID: 0,
        CONFIG_REFINER_MODEL_ID: 0,
        CONFIG_CLIP: 3,
        CONFIG_STYLES: [],
        CONFIG_ENABLE_HR: 0,
        CONFIG_HR_SCALE: 1.55,
        CONFIG_CFG_SCALE: 7.0,
        CONFIG_UPSCALER: 0,
        CONFIG_SAMPLER: 0,
        CONFIG_STEPS: 20,
        CONFIG_DENO_STRENGTH: 0.65,
        CONFIG_ENABLE_TRANSLATE: 0,
        CONFIG_ENABLE_CONTROLNET: 0,
        CONFIG_ENABLE_ADETAILER: 0,
        CONFIG_ENABLE_REFINER: 0,
        CONFIG_ENABLE_SHUFFLE_PROMPT: 0,
        CONFIG_ENABLE_DYNAMIC_PROMPT: 1,
        CONFIG_SEND_BATCH_SIZE: 18,
        CONFIG_UNIT_TIMEOUT: 180,
        CONFIG_SWITCH_AT: 0.7,
        CONFIG_APPEND_LORAS: [],
        # in the current version of QQ transmitting protocol,
        # 20 is the maximum of the pictures that can be sent at once
    }

    @classmethod
    def get_plugin_name(cls) -> str:
        return "StableDiffusionDev"

    @classmethod
    def get_plugin_description(cls) -> str:
        return "a stable diffusion plugin"

    @classmethod
    def get_plugin_version(cls) -> str:
        return "0.2.3"

    @classmethod
    def get_plugin_author(cls) -> str:
        return "whth"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sd_app = StableDiffusionApp(
            host_url=(self._config_registry.get_config(self.CONFIG_SD_HOST)),
            cache_dir=(self._config_registry.get_config(self.CONFIG_IMG_TEMP_DIR_PATH)),
            output_dir=(self._config_registry.get_config(self.CONFIG_OUTPUT_DIR_PATH)),
        )

    def install(self):
        # region local utils
        translater: Optional[AbstractPlugin] = self._plugin_view.get(self.__TRANSLATE_PLUGIN_NAME, None)
        translate: Optional[StableDiffusionPlugin.__TRANSLATE_METHOD_TYPE] = None
        if translater:
            translate = getattr(translater, self.__TRANSLATE_METHOD_NAME)

        cmd_builder = CmdBuilder(
            config_setter=self._config_registry.set_config, config_getter=self._config_registry.get_config
        )
        controlnet_app: Controlnet = Controlnet(host_url=self._config_registry.get_config(self.CONFIG_SD_HOST))

        random_prompt_gen = RandomPromptGenerator(
            wildcard_manager=WildcardManager(path=self._config_registry.get_config(self.CONFIG_WILDCARD_DIR_PATH))
        )
        sd_options = Options(host_url=self.config_registry.get_config(self.CONFIG_SD_HOST))
        processor = PromptProcessorRegistry()

        lora_manager = LoraManager()
        lora_manager.reasign_pool(self.sd_app.available_lora_models)
        lora_manager.parse_container(self.config_registry.get_config(self.CONFIG_APPEND_LORAS))
        processor.register(
            judge=lambda: self._config_registry.get_config(self.CONFIG_ENABLE_DYNAMIC_PROMPT),
            process_engine=lambda prompt: random_prompt_gen.generate(template=prompt)[0] if prompt else "",
            process_name="DYNAMIC_PROMPT_INTERPRET",
        )
        processor.register(
            judge=lambda: self._config_registry.get_config(self.CONFIG_ENABLE_TRANSLATE),
            process_engine=partial(translate, tolang="en", fromlang="auto"),
            process_name="TRANSLATE",
        ) if translate else None
        processor.register(
            judge=lambda: self._config_registry.get_config(self.CONFIG_ENABLE_SHUFFLE_PROMPT),
            process_engine=shuffle_prompt,
            process_name="SHUFFLE",
        )
        processor.register(
            judge=lambda: True,
            process_engine=make_lora_replace_process_engine(self.sd_app.available_lora_models),
            process_name="LORA_REPLACE",
        )
        processor.register(
            judge=lambda: lora_manager.container,
            processor=lambda pos, neg: (pos + lora_manager.dedup().format(), neg),
            process_name="LORA_APPEND",
        )
        processor.register(
            judge=lambda: lora_manager.container,
            process_engine=lambda prompt: prompt.replace("，", ","),
            process_name="CommaReplace",
        )

        configurable_options: Set[str] = {
            self.CONFIG_CURRENT_MODEL_ID,
            self.CONFIG_REFINER_MODEL_ID,
            self.CONFIG_CLIP,
            self.CONFIG_SEND_BATCH_SIZE,
            self.CONFIG_UNIT_TIMEOUT,
            self.CONFIG_ENABLE_HR,
            self.CONFIG_HR_SCALE,
            self.CONFIG_UPSCALER,
            self.CONFIG_DENO_STRENGTH,
            self.CONFIG_ENABLE_TRANSLATE,
            self.CONFIG_ENABLE_DYNAMIC_PROMPT,
            self.CONFIG_ENABLE_SHUFFLE_PROMPT,
            self.CONFIG_ENABLE_CONTROLNET,
            self.CONFIG_ENABLE_ADETAILER,
            self.CONFIG_ENABLE_REFINER,
            self.CONFIG_CONTROLNET_MODULE,
            self.CONFIG_CONTROLNET_MODEL,
            self.CONFIG_SWITCH_AT,
            self.CONFIG_CFG_SCALE,
            self.CONFIG_SAMPLER,
            self.CONFIG_STEPS,
        }

        # endregion

        # region std cmds
        async def diffusion_history_t() -> List[Image]:
            """
            Retrieves the diffusion history and converts it into an image.

            Returns:
                Image: The image representation of the diffusion history.
            """

            return make_image_form_paths(await self.sd_app.txt2img_history())

        async def diffusion_history_i() -> List[Image]:
            """
            Retrieves the diffusion history and converts it into an image.

            Returns:
                Image: The image representation of the diffusion history.
            """
            return make_image_form_paths(await self.sd_app.img2img_history())

        async def diffusion_favorite_t(index: int = None) -> List[Image]:
            """
            An asynchronous function that retrieves a favorite image from the SD_app based on the given index.

            Parameters:
                index (int, optional): The index of the favorite image to retrieve. Defaults to None.

            Returns:
                List[Image]: A list of Image objects containing the retrieved favorite image.

            Example:
                favorite_images = await diffusion_favorite_t(5)
            """
            return make_image_form_paths(await self.sd_app.txt2img_favorite(index))

        async def diffusion_favorite_i(index: int = None) -> List[Image]:
            """
            Asynchronously retrieves the favorite image at a given index from the diffusion app.

            Args:
                index (int, optional): The index of the favorite image to retrieve. Defaults to None.

            Returns:
                List[Image]: A list of Image objects representing the favorite image(s) retrieved.
            """
            return make_image_form_paths(await self.sd_app.img2img_favorite(index))

        def _test_process(pos_prompt: Optional[str] = None, neg_prompt: Optional[str] = None) -> str:
            """
            Process the positive and negative prompts using the provided prompts or default prompts.

            Args:
                pos_prompt (Optional[str]): The positive prompt. Defaults to None.
                neg_prompt (Optional[str]): The negative prompt. Defaults to None.

            Returns:
                str: A string containing the processed positive and negative prompts.
            """
            pos_prompt = pos_prompt or get_default_pos_prompt()
            neg_prompt = neg_prompt or get_default_neg_prompt()
            pos_prompt, neg_prompt = processor.process(pos_prompt, neg_prompt)
            return f"Pos prompt\n----------------\n{pos_prompt}\n\nNeg prompt\n----------------\n{neg_prompt}"

        def _explain() -> str:
            """
            Explain the function and its purpose.

            Returns:
                str: The explanation of the function.
            """
            stdout = ""
            crmodel = self.sd_app.available_sd_models[self.config_registry.get_config(self.CONFIG_CURRENT_MODEL_ID)]
            rfmodel = self.sd_app.available_sd_models[self.config_registry.get_config(self.CONFIG_REFINER_MODEL_ID)]
            upscaler = self.sd_app.available_upscalers[self.config_registry.get_config(self.CONFIG_UPSCALER)]
            sampler: str = self.sd_app.available_samplers[self.config_registry.get_config(self.CONFIG_SAMPLER)]
            stdout += f"Current model:\n{crmodel}\n"
            stdout += f"Refiner model:\n{rfmodel}\n"
            stdout += f"Upscaler:\n{upscaler}\n"
            stdout += f"Sampler:\n{sampler}\n"
            return stdout

        # endregion
        @self.receiver(ApplicationLaunch)
        async def fetch_resources():
            """
            Asynchronous function that fetches resources upon application launch.

            This function is decorated with `@self.receiver(ApplicationLaunch)` to indicate that it is a receiver for
            the `ApplicationLaunch` event.

            The function performs the following tasks:
            - Calls the `fetch_resources()` method of the `controlnet_app` object.
            - Calls the `fetch_config()` method of the `sd_options` object.
            - Calls the `fetch_sd_models()` method of the `self.sd_app` object.
            - Calls the `fetch_lora_models()` method of the `self.sd_app` object.

            Returns:
                None
            """

            async with ClientSession(base_url=self._config_registry.get_config(self.CONFIG_SD_HOST)) as fetch_session:
                await controlnet_app.fetch_resources(fetch_session)
                await sd_options.fetch_config(fetch_session)
                await self.sd_app.fetch_sd_models(fetch_session)
                await self.sd_app.fetch_lora_models(fetch_session)
                await self.sd_app.fetch_upscalers(fetch_session)
                await self.sd_app.fetch_sampler(fetch_session)

        async def rand_lora_generation(count: int = 1) -> MessageChain | str:
            """
            Asynchronously generates a random lora message chain with the specified count.

            Args:
                count (int, optional): The number of lora messages to generate. Defaults to 1.

            Returns:
                MessageChain | str: The generated lora message chain or a string indicating no lora models found.
            """
            pre_append_string = "girl,solo"
            if not self.sd_app.available_lora_models:
                return "No lora models found OR lora models not fetched"

            count = count or 1
            count = 1 if count < 1 else count

            lora_seq_len_sub1: int = len(self.sd_app.available_lora_models) - 1
            chain_seq = MessageChain([])
            for _ in range(count):
                rand_lora = random.randint(0, lora_seq_len_sub1)
                lora_name: str = self.sd_app.available_lora_models[rand_lora]

                prompt, _ = processor.process(pos_prompt=f"lr:{rand_lora},{pre_append_string}", neg_prompt="")

                images = await self.sd_app.txt2img(
                    diffusion_parameters=DiffusionParser(
                        prompt=prompt, styles=self.config_registry.get_config(self.CONFIG_STYLES)
                    ),
                    hires_parameters=HiResParser(  # Get enable HR flag from configuration
                        enable_hr=self._config_registry.get_config(self.CONFIG_ENABLE_HR),
                        hr_scale=self._config_registry.get_config(self.CONFIG_HR_SCALE),
                        hr_upscaler=self.sd_app.available_upscalers[
                            self._config_registry.get_config(self.CONFIG_UPSCALER)
                        ]
                        if self.sd_app.available_upscalers
                        else "",
                        denoising_strength=self._config_registry.get_config(self.CONFIG_DENO_STRENGTH),
                    ),
                )
                chain_seq.append(lora_name)
                chain_seq.append(Image(path=images[0]))

            return chain_seq

        def _add_lora(*tokens: str) -> str:
            """
            A function to add a Lora unit, specifying the index and optional weight.

            Args:
                index (int): The index of the Lora unit.
                weight (float, optional): The weight of the Lora unit. Defaults to 1.0.

            Returns:
                str: The formatted string representing the Lora units.
            """

            converted_tokens = [float(token) if "." in token else int(token) for token in tokens]
            if converted_tokens == 0:
                return _what_lora()

            default_weight = 1.0

            while converted_tokens:
                match converted_tokens.pop(0):
                    case int(index):
                        lora_manager.use(
                            index,
                            converted_tokens.pop(0)
                            if converted_tokens and isinstance(converted_tokens[0], float)
                            else default_weight,
                        )
                    case _:
                        return "Invalid Lora index, float can only be applied when putting after the index"

            return _what_lora()

        def _remove_lora(*indexes: str) -> str:
            """
            A function that removes a Lora unit at the specified index and returns a formatted string of the remaining Lora units.

            :param index: An integer representing the index of the Lora unit to be removed.
            :return: A string containing the formatted Lora units after removal.
            """
            converted_indexes = [int(index) for index in indexes]
            for converted_index in converted_indexes:
                lora_manager.remove(converted_index)
            return _what_lora()

        def _save_lora():
            """

            save lora

            """
            self.config_registry.set_config(self.CONFIG_APPEND_LORAS, lora_manager.dump_container())

            return _what_lora()

        def _what_lora() -> str:
            """
            A function that updates the Lora units index, deduplicates them, and returns a markdown table complex dictionary containing Pool_index, Lora_Units, and Weights.
            """
            lora_manager.update_index().dedup()
            return dict_to_markdown_table_complex(
                {
                    "Index": [unit.index for unit in lora_manager.container],
                    "Name": [unit.name for unit in lora_manager.container],
                    "Weights": [unit.weight for unit in lora_manager.container],
                },
                add_index=False,
            )

        def _set_all_lora_weight(weight: float) -> str:
            """
            Sets the weight of all Lora units to the specified value.

            Args:
                weight (float): The weight value to set for all Lora units.

            Returns:
                str: A formatted string representing the Lora units after updating their weights.

            """
            for unit in lora_manager.container:
                unit.weight = weight
            return _what_lora()

        tree = NameSpaceNode(
            name=CMD.stablediffusion.name,
            aliases=CMD.stablediffusion.value,
            required_permissions=self.required_permission,
            help_message=self.get_plugin_description(),
            children_node=[
                NameSpaceNode(
                    name=CMD.config.name,
                    aliases=CMD.config.value,
                    children_node=[
                        ExecutableNode(
                            name=CMD.list.name,
                            aliases=CMD.list.value,
                            source=cmd_builder.build_list_out_for(configurable_options),
                        ),
                        ExecutableNode(
                            name=CMD.set.name,
                            aliases=CMD.set.value,
                            source=cmd_builder.build_setter_hall(),
                        ),
                        ExecutableNode(
                            name=CMD.explain.name,
                            aliases=CMD.explain.value,
                            source=_explain,
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.controlnet.name,
                    aliases=CMD.controlnet.value,
                    children_node=[
                        ExecutableNode(
                            name=CMD.models.name,
                            aliases=CMD.models.value,
                            source=lambda: make_stdout_seq_string(controlnet_app.models, title="CN_Models"),
                        ),
                        ExecutableNode(
                            name=CMD.modules.name,
                            aliases=CMD.modules.value,
                            source=lambda: make_stdout_seq_string(controlnet_app.modules, title="CN_Modules"),
                        ),
                        ExecutableNode(
                            name=CMD.detect.name,
                            aliases=CMD.detect.value,
                            help_message="Detect a image use controlnet",
                            source=lambda: None,
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.again.name,
                    aliases=CMD.again.value,
                    help_message="Generate from history, img2img or txt2img",
                    children_node=[
                        ExecutableNode(
                            name=CMD.txt2img.name,
                            aliases=CMD.txt2img.value,
                            source=diffusion_history_t,
                        ),
                        ExecutableNode(
                            name=CMD.img2img.name,
                            aliases=CMD.img2img.value,
                            source=diffusion_history_i,
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.like.name,
                    aliases=CMD.like.value,
                    help_message="mark last generation as liked,or generate from favorite",
                    children_node=[
                        NameSpaceNode(
                            name=CMD.size.name,
                            aliases=CMD.size.value,
                            help_message="the size of the Favorite storage",
                            children_node=[
                                ExecutableNode(
                                    name=CMD.txt2img.name,
                                    aliases=CMD.txt2img.value,
                                    source=lambda: f"The size of the t2i Favorite storage is: \n"
                                    f"{len(self.sd_app.txt2img_params.favorite)}",
                                ),
                                ExecutableNode(
                                    name=CMD.img2img.name,
                                    aliases=CMD.img2img.value,
                                    source=lambda: f"The size of the i2i Favorite storage is: \n"
                                    f"{len(self.sd_app.img2img_params.favorite)}",
                                ),
                            ],
                        ),
                        ExecutableNode(
                            name=CMD.txt2img.name,
                            aliases=CMD.txt2img.value,
                            source=lambda: f"add to t2i favorite\nSuccess={self.sd_app.add_favorite_t()}",
                        ),
                        ExecutableNode(
                            name=CMD.img2img.name,
                            aliases=CMD.img2img.value,
                            source=lambda: f"add to i2i favorite\nSuccess={self.sd_app.add_favorite_i()}",
                        ),
                        NameSpaceNode(
                            name=CMD.again.name,
                            aliases=CMD.again.value,
                            help_message="retrieve a favorite generation, with index or random",
                            children_node=[
                                ExecutableNode(
                                    name=CMD.txt2img.name,
                                    aliases=CMD.txt2img.value,
                                    source=diffusion_favorite_t,
                                ),
                                ExecutableNode(
                                    name=CMD.img2img.name,
                                    aliases=CMD.img2img.value,
                                    source=diffusion_favorite_i,
                                ),
                            ],
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.default.name,
                    aliases=CMD.default.value,
                    help_message="Set default settings for the plugin",
                    children_node=[
                        ExecutableNode(
                            name=CMD.positive.name,
                            aliases=CMD.positive.value,
                            source=lambda x: f"Set pos prompt to\n{x}\n\nSuccess={set_default_pos_prompt(x)}",
                        ),
                        ExecutableNode(
                            name=CMD.negative.name,
                            aliases=CMD.negative.value,
                            source=lambda x: f"Set neg prompt to\n{x}\nSuccess={set_default_neg_prompt(x)}",
                        ),
                        ExecutableNode(
                            name=CMD.shot.name,
                            aliases=CMD.shot.value,
                            source=lambda x: f"Set shot size\nSuccess={set_shot_size(x)}",
                        ),
                    ],
                ),
                ExecutableNode(
                    name=CMD.test.name,
                    aliases=CMD.test.value,
                    source=_test_process,
                ),
                NameSpaceNode(
                    name=CMD.models.name,
                    aliases=CMD.models.value,
                    help_message="get available models,StableDiffusion/Lora",
                    children_node=[
                        ExecutableNode(
                            name=CMD.lora.name,
                            aliases=CMD.lora.value,
                            help_message="get available lora models",
                            source=lambda: dict_to_markdown_table_complex(
                                {"Loras": [s.split(".")[0] for s in self.sd_app.available_lora_models]}
                            ),
                        ),
                        ExecutableNode(
                            name=CMD.normal.name,
                            aliases=CMD.normal.value,
                            help_message="get available stable diffusion models",
                            source=lambda: dict_to_markdown_table_complex(
                                {"SD_Models": [s.split(".")[0] for s in self.sd_app.available_sd_models]}
                            ),
                        ),
                        ExecutableNode(
                            name=CMD.upscaler.name,
                            aliases=CMD.upscaler.value,
                            help_message="get available upscalers",
                            source=lambda: dict_to_markdown_table_complex(
                                {"Upscalers": self.sd_app.available_upscalers}
                            ),
                        ),
                        ExecutableNode(
                            **CMD.sampler.export(),
                            help_message="get available samplers",
                            source=lambda: dict_to_markdown_table_complex({"Samplers": self.sd_app.available_samplers}),
                        ),
                    ],
                ),
                ExecutableNode(
                    name=CMD.interrogate.name,
                    aliases=CMD.interrogate.value,
                    help_message="Interrogate the image content",
                    source=lambda x: None,
                ),
                ExecutableNode(name=CMD.fetch.name, aliases=CMD.fetch.value, source=fetch_resources),
                ExecutableNode(name=CMD.halt.name, aliases=CMD.halt.value, source=lambda: self.sd_app.interrupt()),
                ExecutableNode(**CMD.randlora.export(), source=rand_lora_generation),
                ExecutableNode(**CMD.yield_lora.export(), source=_add_lora),
                ExecutableNode(**CMD.remove_lora.export(), source=_remove_lora),
                ExecutableNode(
                    **CMD.clean_lora.export(), source=lambda: f"Clean success = {bool(lora_manager.clean().container)}"
                ),
                ExecutableNode(**CMD.what_lora.export(), source=_what_lora),
                ExecutableNode(**CMD.save_lora.export(), source=_save_lora),
                ExecutableNode(**CMD.weight.export(), source=_set_all_lora_weight),
            ],
        )

        self._root_namespace_node.add_node(tree)

        # region castings
        @self.receiver(
            [FriendMessage, GroupMessage],  # List of message types that this function can handle
            decorators=[
                ContainKeyword(keyword=keyword)
                for keyword in self._config_registry.get_config(
                    self.CONFIG_POS_KEYWORD
                )  # List comprehension to generate decorators based on configuration
            ],
        )
        async def diffusion(
            app: Ariadne,  # The Ariadne instance
            target: Union[Group, Friend],  # The target group or friend to send the image to
            message: MessageChain,  # The message containing the prompts for diffusion
            message_event: Union[GroupMessage, FriendMessage],  # The message event
        ):
            """
            Asynchronously performs diffusion on the given message and sends the resulting image as a message
            in the group or to a friend.

            Args:
                app (Ariadne): The Ariadne instance.
                target (Union[Group, Friend]): The target group or friend to send the image to.
                message (MessageChain): The message containing the prompts for diffusion.
                message_event (Union[GroupMessage, FriendMessage]): The message event.

            Returns:
                None
            """
            # Extract positive and negative prompts from the message
            try:
                extracted_pos_prompt, extracted_neg_prompt, batch_count = extract_prompts(
                    str(message),
                    pos_keyword=self._config_registry.get_config(
                        self.CONFIG_POS_KEYWORD
                    ),  # Get positive keyword from configuration
                    neg_keyword=self._config_registry.get_config(
                        self.CONFIG_NEG_KEYWORD
                    ),  # Get negative keyword from configuration
                    raise_settings=(True, False, False),
                )
            except ValueError as e:
                print(e)
                return

            pos_prompt = extracted_pos_prompt or get_default_pos_prompt()  # Get positive prompt or use default prompt
            neg_prompt = extracted_neg_prompt or get_default_neg_prompt()  # Get negative prompt or use default prompt

            image_url = await get_image_url(app, message_event)  # Get image URL from message event
            send_result = []
            send_batch_size = self.config_registry.get_config(
                self.CONFIG_SEND_BATCH_SIZE
            )  # Get send batch size from configuration
            overrides: None | OverRideSettings = None
            ref_parser: None | RefinerParser = None
            if self.sd_app.available_sd_models:
                sd_options.record_start()
                sd_options.sd_model_checkpoint = self.sd_app.available_sd_models[
                    self.config_registry.get_config(self.CONFIG_CURRENT_MODEL_ID)
                ]
                sd_options.CLIP_stop_at_last_layers = self.config_registry.get_config(self.CONFIG_CLIP)

                overrides = sd_options.generate_override_settings_payload(
                    sd_options.record_end(), recover_after_override=False
                )
                ref_parser = (
                    RefinerParser(
                        refiner_checkpoint=(
                            self.sd_app.available_sd_models[
                                self.config_registry.get_config(self.CONFIG_REFINER_MODEL_ID)
                            ]
                        ),
                        refiner_switch_at=self.config_registry.get_config(self.CONFIG_SWITCH_AT),
                    )
                    if self.config_registry.get_config(self.CONFIG_ENABLE_REFINER)
                    else None
                )

            async with ClientSession(
                base_url=self.config_registry.get_config(self.CONFIG_SD_HOST),
                timeout=ClientTimeout(total=self.config_registry.get_config(self.CONFIG_UNIT_TIMEOUT) * batch_count),
            ) as session:
                for _ in range(batch_count):
                    final_pos_prompt, final_neg_prompt = processor.process(pos_prompt, neg_prompt)  # Process prompts
                    # Create a diffusion parser with the prompts
                    diffusion_parser = DiffusionParser(
                        prompt=final_pos_prompt,
                        negative_prompt=final_neg_prompt,
                        styles=self._config_registry.get_config(self.CONFIG_STYLES),  # Get styles from configuration
                        sampler_name=self.sd_app.available_samplers[
                            self.config_registry.get_config(self.CONFIG_SAMPLER)
                        ],
                        cfg_scale=self.config_registry.get_config(
                            self.CONFIG_CFG_SCALE
                        ),  # Get CFG scale from configuration
                        steps=self.config_registry.get_config(self.CONFIG_STEPS),
                    )

                    adetailer_parser = (
                        ADetailerArgs(
                            ad_unit=[
                                ADetailerUnit(
                                    ad_model=ModelType.FACE_YOLOV8M.value,
                                ),
                                ADetailerUnit(
                                    ad_model=ModelType.PERSON_YOLOV8MSEG.value,
                                ),
                            ]
                        )
                        if self.config_registry.get_config(self.CONFIG_ENABLE_ADETAILER)
                        else None
                    )
                    if image_url:
                        send_result.extend(
                            await _make_img2img(
                                diffusion_parser, image_url, override_settings=overrides, refiner_parameters=ref_parser
                            )
                        )  # Make image-to-image diffusion
                    else:
                        # Generate the image using the diffusion parser

                        send_result.extend(
                            await self.sd_app.txt2img(
                                diffusion_parameters=diffusion_parser,
                                hires_parameters=HiResParser(  # Get enable HR flag from configuration
                                    enable_hr=self._config_registry.get_config(self.CONFIG_ENABLE_HR),
                                    hr_scale=self._config_registry.get_config(self.CONFIG_HR_SCALE),
                                    hr_upscaler=self.sd_app.available_upscalers[
                                        self._config_registry.get_config(self.CONFIG_UPSCALER)
                                    ]
                                    if self.sd_app.available_upscalers
                                    else "",
                                    denoising_strength=self._config_registry.get_config(self.CONFIG_DENO_STRENGTH),
                                ),
                                refiner_parameters=ref_parser,
                                adetailer_parameters=adetailer_parser,
                                override_settings=overrides,
                                session=session,
                            )
                        )

                    # Split the send_result into batches, whose size is send_batch_size
                    split_batch = split_list(
                        input_list=send_result, sublist_size=send_batch_size, strip_remains=False
                    )  # Split send_result into batches
                    send_batches, send_result = split_batch[:-1], split_batch[-1]
                    for batch in send_batches:
                        # Send the image as a message in the group
                        await app.send_message(target, [Image(path=path) for path in batch])

            if send_result:
                # deal with the remains
                await app.send_message(target, [Image(path=path) for path in send_result])

        @self.receiver(
            [FriendMessage, GroupMessage],
            decorators=[
                MatchRegex(
                    regex=assemble_cmd_regex_parts(
                        [
                            make_regex_part_from_enum(CMD.stablediffusion),
                            make_regex_part_from_enum(CMD.controlnet),
                            make_regex_part_from_enum(CMD.detect),
                        ]
                    )
                    + ".*"
                )
            ],
        )
        async def cn_detect(
            app: Ariadne,
            target: Union[Group, Friend],
            message: MessageChain,
        ) -> None:
            """
            Detects the controlnet in the given message.
            """
            if Image not in message:
                return
            images = message.get(Image)
            print(f"Detecting images count: {len(images)}\n")
            pay_load = ControlNetDetect(
                controlnet_module=self._config_registry.get_config(self.CONFIG_CONTROLNET_MODULE),
                controlnet_input_images=[
                    img_to_base64(
                        await download_file(
                            img.url, save_dir=self._config_registry.get_config(self.CONFIG_IMG_TEMP_DIR_PATH)
                        )
                    )
                    for img in images
                ],
            )
            img_base64_list = await controlnet_app.detect(payload=pay_load)
            if not img_base64_list:
                await app.send_message(target, "Something went wrong, please try again later.")
                return None
            await app.send_message(target, [Image(base64=img_base64) for img_base64 in img_base64_list])

        @self.receiver(
            [FriendMessage, GroupMessage],
            decorators=[
                MatchRegex(
                    regex=assemble_cmd_regex_parts(
                        [
                            make_regex_part_from_enum(CMD.stablediffusion),
                            make_regex_part_from_enum(CMD.interrogate),
                        ]
                    )
                    + ".*"
                )
            ],
        )
        async def interrogate(
            app: Ariadne,
            target: Union[Group, Friend],
            message: MessageChain,
        ) -> None:
            """
            Interrogates the given message.
            """
            if Image not in message:
                return
            images = message.get(Image)
            if not images:
                return
            image = images[0]
            print(f"Interrogating images count: {len(images)}\n")

            await app.send_message(
                target,
                "\n".join(
                    f"{prob:<5}=> {tag}"
                    for tag, prob in (
                        await self.sd_app.interrogate_image(
                            parser=InterrogateParser(
                                image_path=await download_file(
                                    image.url,
                                    save_dir=self.config_registry.get_config(self.CONFIG_IMG_TEMP_DIR_PATH),
                                )
                            )
                        )
                    ).items()
                ),
            )

        async def _make_img2img(
            diffusion_paser: DiffusionParser,
            image_url: str,
            override_settings: OverRideSettings,
            refiner_parameters: RefinerParser = None,
        ) -> List[str]:
            # Download the first image in the chain
            print(f"Downloading image from: {image_url}\n")
            img_path = await download_file(
                image_url,
                save_dir=self._config_registry.get_config(self.CONFIG_IMG_TEMP_DIR_PATH),
            )
            img_base64 = img_to_base64(img_path)
            cn_unit = None
            if self._config_registry.get_config(self.CONFIG_ENABLE_CONTROLNET):
                # ControlNet
                module: str = self._config_registry.get_config(self.CONFIG_CONTROLNET_MODULE)
                model: str = self._config_registry.get_config(self.CONFIG_CONTROLNET_MODEL)
                print(f"Using ControlNet module: {module}, model: {model}\n")
                if module in controlnet_app.modules and model in controlnet_app.models:
                    cn_unit = ControlNetUnit(
                        input_image=img_base64,
                        module=module,
                        model=model,
                    )
            if self.config_registry.get_config(self.CONFIG_ENABLE_HR):
                hr_scale = self.config_registry.get_config(self.CONFIG_HR_SCALE)
                diffusion_paser.width = int(diffusion_paser.width * hr_scale)
                diffusion_paser.height = int(diffusion_paser.height * hr_scale)
            send_result = await self.sd_app.img2img(
                diffusion_parameters=diffusion_paser,
                controlnet_parameters=cn_unit,
                image_base64=img_base64,
                override_settings=override_settings,
                refiner_parameters=refiner_parameters,
            )
            return send_result

        # endregion

    async def interrogate(self, image: str) -> OrderedDict[str, float]:
        """
        Interrogate an image and return the result as a dictionary.

        Args:
            image (str): The path to an image or a base64 encoded image.

        Returns:
            dict: The result of the interrogation.
        """
        # Check if the image path exists
        if pathlib.Path(image).exists():
            # Interrogate the image using the image path
            return await self.sd_app.interrogate_image(parser=InterrogateParser(image_path=image))
        else:
            # Interrogate the image using the base64 encoded image
            return await self.sd_app.interrogate_image(parser=InterrogateParser(image=image))
