import pathlib
from functools import partial
from typing import List, Optional, Callable, Union, OrderedDict

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
)
from .adetailer import ADetailerArgs, ADetailerUnit, ModelType
from .controlnet import ControlNetUnit, Controlnet, ControlNetDetect
from .extractors import get_image_url, make_image_form_paths
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


class CMD:
    ROOT = "sd"
    AGAIN = "ag"
    IMG2IMG = "i"
    TXT2IMG = "t"
    LIKE = "lk"
    SIZE = "size"
    DEFAULT = "default"
    SET = "set"
    CONFIG = "config"
    LIST_OUT = "list"

    MODELS = "models"
    MODULES = "modules"

    INTERROGATE = "expl"
    NORMAL_MODEL = "sd"
    LORA_MODEL = "lora"
    UPSCALER = "ups"

    POS_PROMPT = "pos"
    NEG_PROMPT = "neg"
    SHOT_SIZE = "shot"
    TEST = "test"

    CONTROLNET = "cn"
    CONTROLNET_DETECT = "d"

    FETCH: str = "fetch"
    HALT: str = "halt"


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
        CONFIG_UPSCALER: 0,
        CONFIG_DENO_STRENGTH: 0.65,
        CONFIG_ENABLE_TRANSLATE: 0,
        CONFIG_ENABLE_CONTROLNET: 0,
        CONFIG_ENABLE_ADETAILER: 0,
        CONFIG_ENABLE_REFINER: 0,
        CONFIG_ENABLE_SHUFFLE_PROMPT: 0,
        CONFIG_ENABLE_DYNAMIC_PROMPT: 1,
        CONFIG_SEND_BATCH_SIZE: 18,
        CONFIG_UNIT_TIMEOUT: 180,
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
        return "0.2.2"

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

        configurable_options: List[str] = [
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
        ]

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

        # endregion
        @self.receiver(ApplicationLaunch)
        async def fetch_resources():
            """
            Asynchronous function that fetches resources upon application launch.

            This function is decorated with `@self.receiver(ApplicationLaunch)` to indicate that it is a receiver for the `ApplicationLaunch` event.

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

        tree = NameSpaceNode(
            name=CMD.ROOT,
            required_permissions=self.required_permission,
            help_message=self.get_plugin_description(),
            children_node=[
                NameSpaceNode(
                    name=CMD.CONFIG,
                    children_node=[
                        ExecutableNode(
                            name=CMD.LIST_OUT,
                            source=cmd_builder.build_list_out_for(configurable_options),
                        ),
                        ExecutableNode(
                            name=CMD.SET,
                            source=cmd_builder.build_setter_hall(),
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.CONTROLNET,
                    children_node=[
                        ExecutableNode(
                            name=CMD.MODELS,
                            source=lambda: make_stdout_seq_string(controlnet_app.models, title="CN_Models"),
                        ),
                        ExecutableNode(
                            name=CMD.MODULES,
                            source=lambda: make_stdout_seq_string(controlnet_app.modules, title="CN_Modules"),
                        ),
                        ExecutableNode(
                            name=CMD.CONTROLNET_DETECT,
                            help_message="Detect a image use controlnet",
                            source=lambda: None,
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.AGAIN,
                    help_message="Generate from history, img2img or txt2img",
                    children_node=[
                        ExecutableNode(
                            name=CMD.TXT2IMG,
                            help_message=diffusion_history_t.__doc__,
                            source=diffusion_history_t,
                        ),
                        ExecutableNode(
                            name=CMD.IMG2IMG,
                            help_message=diffusion_history_i.__doc__,
                            source=diffusion_history_i,
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.LIKE,
                    help_message="mark last generation as liked,or generate from favorite",
                    children_node=[
                        NameSpaceNode(
                            name=CMD.SIZE,
                            help_message="the size of the Favorite storage",
                            children_node=[
                                ExecutableNode(
                                    name=CMD.TXT2IMG,
                                    source=lambda: f"The size of the t2i Favorite storage is: \n"
                                    f"{len(self.sd_app.txt2img_params.favorite)}",
                                ),
                                ExecutableNode(
                                    name=CMD.IMG2IMG,
                                    source=lambda: f"The size of the i2i Favorite storage is: \n"
                                    f"{len(self.sd_app.img2img_params.favorite)}",
                                ),
                            ],
                        ),
                        ExecutableNode(
                            name=CMD.TXT2IMG,
                            source=lambda: f"add to t2i favorite\nSuccess={self.sd_app.add_favorite_t()}",
                        ),
                        ExecutableNode(
                            name=CMD.IMG2IMG,
                            source=lambda: f"add to i2i favorite\nSuccess={self.sd_app.add_favorite_i()}",
                        ),
                        NameSpaceNode(
                            name=CMD.AGAIN,
                            help_message="retrieve a favorite generation, with index or random",
                            children_node=[
                                ExecutableNode(
                                    name=CMD.TXT2IMG,
                                    help_message=diffusion_favorite_t.__doc__,
                                    source=diffusion_favorite_t,
                                ),
                                ExecutableNode(
                                    name=CMD.IMG2IMG,
                                    help_message=diffusion_favorite_i.__doc__,
                                    source=diffusion_favorite_i,
                                ),
                            ],
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.DEFAULT,
                    help_message="Set default settings for the plugin",
                    children_node=[
                        ExecutableNode(
                            name=CMD.POS_PROMPT,
                            help_message=set_default_pos_prompt.__doc__,
                            source=lambda x: f"Set pos prompt to\n{x}\n\nSuccess={set_default_pos_prompt(x)}",
                        ),
                        ExecutableNode(
                            name=CMD.NEG_PROMPT,
                            help_message=set_default_neg_prompt.__doc__,
                            source=lambda x: f"Set neg prompt to\n{x}\nSuccess={set_default_neg_prompt(x)}",
                        ),
                        ExecutableNode(
                            name=CMD.SHOT_SIZE,
                            help_message=set_shot_size.__doc__,
                            source=lambda x: f"Set shot size\nSuccess={set_shot_size(x)}",
                        ),
                    ],
                ),
                ExecutableNode(
                    name=CMD.TEST,
                    help_message=_test_process.__doc__,
                    source=_test_process,
                ),
                NameSpaceNode(
                    name=CMD.MODELS,
                    help_message="get available models,StableDiffusion/Lora",
                    children_node=[
                        ExecutableNode(
                            name=CMD.LORA_MODEL,
                            help_message="get available lora models",
                            source=lambda: make_stdout_seq_string(
                                seq=self.sd_app.available_lora_models, title="Lora Models"
                            ),
                        ),
                        ExecutableNode(
                            name=CMD.NORMAL_MODEL,
                            help_message="get available stable diffusion models",
                            source=lambda: make_stdout_seq_string(
                                seq=self.sd_app.available_sd_models, title="StableDiffusion Models"
                            ),
                        ),
                        ExecutableNode(
                            name=CMD.UPSCALER,
                            help_message="get available upscalers",
                            source=lambda: make_stdout_seq_string(
                                seq=self.sd_app.available_upscalers, title="Upscalers"
                            ),
                        ),
                    ],
                ),
                ExecutableNode(
                    name=CMD.INTERROGATE, help_message="Interrogate the image content", source=lambda x: None
                ),
                ExecutableNode(name=CMD.FETCH, help_message=fetch_resources.__doc__, source=fetch_resources),
                ExecutableNode(
                    name=CMD.HALT, help_message=self.sd_app.interrupt.__doc__, source=lambda: self.sd_app.interrupt()
                ),
            ],
        )

        self._root_namespace_node.add_node(tree)

        processor.register(
            judge=lambda: self._config_registry.get_config(self.CONFIG_ENABLE_DYNAMIC_PROMPT),
            process_engine=lambda prompt: random_prompt_gen.generate(template=prompt)[0] or prompt,
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
                        )
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
            decorators=[MatchRegex(regex=rf"^{CMD.ROOT}\s+{CMD.CONTROLNET}\s+{CMD.CONTROLNET_DETECT}.*")],
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
                            save_dir=self._config_registry.get_config(self.CONFIG_IMG_TEMP_DIR_PATH), url=img.url
                        )
                    )
                    for img in images
                ],
            )
            img_base64_list = await controlnet_app.detect(payload=pay_load)

            await app.send_message(target, [Image(base64=img_base64) for img_base64 in img_base64_list])

        @self.receiver(
            [FriendMessage, GroupMessage],
            decorators=[MatchRegex(regex=rf"^{CMD.ROOT}\s+{CMD.INTERROGATE}.*")],
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
                save_dir=self._config_registry.get_config(self.CONFIG_IMG_TEMP_DIR_PATH), url=image_url
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
