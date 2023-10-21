import os
from typing import Tuple, List, Optional, Callable, Union

from modules.file_manager import download_file, get_pwd
from modules.plugin_base import AbstractPlugin

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

    POS_PROMPT = "pos"
    NEG_PROMPT = "neg"
    SHOT_SIZE = "shot"
    TEST = "test"
    CONTROLNET_CMD = "cn"
    CONTROLNET_MODELS_CMD = "models"
    CONTROLNET_MODULES_CMD = "modules"
    CONTROLNET_DETECT_CMD = "d"
    # TODO add cn detect cmd


class StableDiffusionPlugin(AbstractPlugin):
    __TRANSLATE_PLUGIN_NAME: str = "BaiduTranslater"
    __TRANSLATE_METHOD_NAME: str = "translate"
    __TRANSLATE_METHOD_TYPE = Callable[[str, str, str], str]  # [tolang, query, fromlang] -> str
    # TODO deal with this super high coupling

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
    CONFIG_STYLES = "styles"
    CONFIG_ENABLE_HR = "enable_hr"

    CONFIG_ENABLE_TRANSLATE = "enable_translate"
    CONFIG_ENABLE_CONTROLNET = "enable_controlnet"
    CONFIG_ENABLE_SHUFFLE_PROMPT = "enable_shuffle_prompt"
    CONFIG_ENABLE_DYNAMIC_PROMPT = "enable_dynamic_prompt"

    CONFIG_SEND_BATCH_SIZE = "send_batch_size"
    DefaultConfig = {
        CONFIG_POS_KEYWORD: "+",
        CONFIG_NEG_KEYWORD: "-",
        CONFIG_OUTPUT_DIR_PATH: f"{get_pwd()}/output",
        CONFIG_IMG_TEMP_DIR_PATH: f"{get_pwd()}/temp",
        CONFIG_SD_HOST: "http://localhost:7860",
        CONFIG_WILDCARD_DIR_PATH: f"{get_pwd()}/asset/wildcard",
        CONFIG_CONTROLNET_MODULE: "openpose_full",
        CONFIG_CONTROLNET_MODEL: "control_v11p_sd15_openpose",
        CONFIG_STYLES: [],
        CONFIG_ENABLE_HR: 0,
        CONFIG_ENABLE_TRANSLATE: 0,
        CONFIG_ENABLE_CONTROLNET: 0,
        CONFIG_ENABLE_SHUFFLE_PROMPT: 0,
        CONFIG_ENABLE_DYNAMIC_PROMPT: 1,
        CONFIG_SEND_BATCH_SIZE: 20,  # in current version of QQ, 20 is the maximum of the pictures that can be sent in a single message
    }

    # TODO this should be removed, use pos prompt keyword and neg prompt keyword
    @classmethod
    def _get_config_dir(cls) -> str:
        return os.path.abspath(os.path.dirname(__file__))

    @classmethod
    def get_plugin_name(cls) -> str:
        return "StableDiffusionDev"

    @classmethod
    def get_plugin_description(cls) -> str:
        return "a stable diffusion plugin"

    @classmethod
    def get_plugin_version(cls) -> str:
        return "0.1.3"

    @classmethod
    def get_plugin_author(cls) -> str:
        return "whth"

    def install(self):
        from graia.ariadne.message.chain import MessageChain, Image
        from graia.ariadne.message.parser.base import ContainKeyword, MatchRegex
        from graia.ariadne.event.message import GroupMessage
        from graia.ariadne.event.lifecycle import ApplicationLaunch
        from graia.ariadne.model import Group
        from dynamicprompts.wildcards import WildcardManager
        from dynamicprompts.generators import RandomPromptGenerator
        from modules.cmd import RequiredPermission, ExecutableNode, NameSpaceNode
        from modules.auth.resources import required_perm_generator
        from modules.auth.permissions import Permission, PermissionCode

        from modules.cmd import CmdBuilder
        from modules.file_manager import img_to_base64
        from .controlnet import ControlNetUnit, Controlnet, ControlNetDetect
        from .stable_diffusion import StableDiffusionApp, DiffusionParser, HiResParser
        from .parser import (
            set_default_pos_prompt,
            set_default_neg_prompt,
            set_shot_size,
            get_default_neg_prompt,
            get_default_pos_prompt,
        )

        from .utils import extract_prompts, PromptProcessorRegistry

        cmd_builder = CmdBuilder(
            config_setter=self._config_registry.set_config, config_getter=self._config_registry.get_config
        )
        controlnet: Controlnet = Controlnet(host_url=self._config_registry.get_config(self.CONFIG_SD_HOST))

        gen = RandomPromptGenerator(
            wildcard_manager=WildcardManager(path=self._config_registry.get_config(self.CONFIG_WILDCARD_DIR_PATH))
        )
        processor = PromptProcessorRegistry()
        configurable_options: List[str] = [
            self.CONFIG_ENABLE_HR,
            self.CONFIG_ENABLE_TRANSLATE,
            self.CONFIG_ENABLE_DYNAMIC_PROMPT,
            self.CONFIG_ENABLE_SHUFFLE_PROMPT,
            self.CONFIG_ENABLE_CONTROLNET,
            self.CONFIG_CONTROLNET_MODULE,
            self.CONFIG_CONTROLNET_MODEL,
        ]

        su_perm = Permission(id=PermissionCode.SuperPermission.value, name=self.get_plugin_name())
        req_perm: RequiredPermission = required_perm_generator(
            target_resource_name=self.get_plugin_name(), super_permissions=[su_perm]
        )

        async def diffusion_history_t() -> Image:
            """
            Retrieves the diffusion history and converts it into an image.

            Returns:
                Image: The image representation of the diffusion history.
            """
            # Retrieve the diffusion history and store the result
            send_result = await SD_app.txt2img_history(output_dir_path)

            # Create an Image object with the path to the result
            return Image(path=send_result[0])

        async def diffusion_history_i() -> Image:
            """
            Retrieves the diffusion history and converts it into an image.

            Returns:
                Image: The image representation of the diffusion history.
            """
            # Retrieve the diffusion history and store the result
            send_result = await SD_app.img2img_history(output_dir_path)

            # Create an Image object with the path to the result
            return Image(path=send_result[0])

        async def diffusion_favorite_t(index: int = None) -> Image:
            images = await SD_app.txt2img_favorite(output_dir_path, index)
            return Image(path=images[0])

        async def diffusion_favorite_i(index: int = None) -> Image:
            images = await SD_app.img2img_favorite(output_dir_path, index)
            return Image(path=images[0])

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

        tree = NameSpaceNode(
            name=CMD.ROOT,
            required_permissions=req_perm,
            help_message=self.get_plugin_description(),
            children_node=[
                NameSpaceNode(
                    name=CMD.CONFIG,
                    required_permissions=req_perm,
                    children_node=[
                        ExecutableNode(
                            name=CMD.LIST_OUT,
                            required_permissions=req_perm,
                            source=cmd_builder.build_list_out_for(configurable_options),
                        ),
                        ExecutableNode(
                            name=CMD.SET,
                            required_permissions=req_perm,
                            source=cmd_builder.build_setter_hall(),
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.CONTROLNET_CMD,
                    required_permissions=req_perm,
                    children_node=[
                        ExecutableNode(
                            name=CMD.CONTROLNET_MODELS_CMD,
                            required_permissions=req_perm,
                            source=lambda: "CN_Models:\n" + "\n".join(controlnet.models),
                        ),
                        ExecutableNode(
                            name=CMD.CONTROLNET_MODULES_CMD,
                            required_permissions=req_perm,
                            source=lambda: "CN_Modules:\n" + "\n".join(controlnet.modules),
                        ),
                        ExecutableNode(
                            name=CMD.CONTROLNET_DETECT_CMD,
                            help_message="Detect a image use controlnet",
                            source=lambda: None,
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.AGAIN,
                    required_permissions=req_perm,
                    help_message="Generate from history, img2img or txt2img",
                    children_node=[
                        ExecutableNode(
                            name=CMD.TXT2IMG,
                            required_permissions=req_perm,
                            source=diffusion_history_t,
                        ),
                        ExecutableNode(
                            name=CMD.IMG2IMG,
                            required_permissions=req_perm,
                            source=diffusion_history_i,
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.LIKE,
                    required_permissions=req_perm,
                    help_message="mark last generation as liked,or generate from favorite",
                    children_node=[
                        NameSpaceNode(
                            name=CMD.SIZE,
                            required_permissions=req_perm,
                            help_message="the size of the Favorite storage",
                            children_node=[
                                ExecutableNode(
                                    name=CMD.TXT2IMG,
                                    required_permissions=req_perm,
                                    source=lambda: f"The size of the t2i Favorite storage is: \n"
                                    f"{len(SD_app.txt2img_params.favorite)}",
                                ),
                                ExecutableNode(
                                    name=CMD.IMG2IMG,
                                    required_permissions=req_perm,
                                    source=lambda: f"The size of the i2i Favorite storage is: \n"
                                    f"{len(SD_app.img2img_params.favorite)}",
                                ),
                            ],
                        ),
                        ExecutableNode(
                            name=CMD.TXT2IMG,
                            required_permissions=req_perm,
                            source=lambda: f"add to t2i favorite\nSuccess={SD_app.add_favorite_t()}",
                        ),
                        ExecutableNode(
                            name=CMD.IMG2IMG,
                            required_permissions=req_perm,
                            source=lambda: f"add to i2i favorite\nSuccess={SD_app.add_favorite_i()}",
                        ),
                        NameSpaceNode(
                            name=CMD.AGAIN,
                            required_permissions=req_perm,
                            help_message="retrieve a favorite generation, with index or random",
                            children_node=[
                                ExecutableNode(
                                    name=CMD.TXT2IMG,
                                    required_permissions=req_perm,
                                    source=diffusion_favorite_t,
                                ),
                                ExecutableNode(
                                    name=CMD.IMG2IMG,
                                    required_permissions=req_perm,
                                    source=diffusion_favorite_i,
                                ),
                            ],
                        ),
                    ],
                ),
                NameSpaceNode(
                    name=CMD.DEFAULT,
                    required_permissions=req_perm,
                    help_message="Set default settings for the plugin",
                    children_node=[
                        ExecutableNode(
                            name=CMD.POS_PROMPT,
                            required_permissions=req_perm,
                            help_message=set_default_pos_prompt.__doc__,
                            source=lambda x: f"Set pos prompt to\n{x}\n\nSuccess={set_default_pos_prompt(x)}",
                        ),
                        ExecutableNode(
                            name=CMD.NEG_PROMPT,
                            required_permissions=req_perm,
                            help_message=set_default_neg_prompt.__doc__,
                            source=lambda x: f"Set neg prompt to\n{x}\nSuccess={set_default_neg_prompt(x)}",
                        ),
                        ExecutableNode(
                            name=CMD.SHOT_SIZE,
                            required_permissions=req_perm,
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
            ],
        )
        self._auth_manager.add_perm_from_req(req_perm)
        self._root_namespace_node.add_node(tree)

        translater: Optional[AbstractPlugin] = self._plugin_view.get(self.__TRANSLATE_PLUGIN_NAME, None)
        if translater:
            translate: StableDiffusionPlugin.__TRANSLATE_METHOD_TYPE = getattr(translater, self.__TRANSLATE_METHOD_NAME)
        output_dir_path = self._config_registry.get_config(self.CONFIG_OUTPUT_DIR_PATH)
        temp_dir_path = self._config_registry.get_config(self.CONFIG_IMG_TEMP_DIR_PATH)

        SD_app = StableDiffusionApp(
            host_url=(self._config_registry.get_config(self.CONFIG_SD_HOST)), cache_dir=temp_dir_path
        )
        self.receiver(ApplicationLaunch)(controlnet.fetch_resources)

        def _dynamic_process(pos_prompt: str, neg_prompt: str) -> Tuple[str, str]:
            pos_interpreted = gen.generate(template=pos_prompt)
            neg_interpreted = gen.generate(template=neg_prompt)
            pos_prompt = pos_interpreted[0] if pos_interpreted else pos_prompt
            neg_prompt = neg_interpreted[0] if neg_interpreted else neg_prompt
            return pos_prompt, neg_prompt

        processor.register(
            judge=lambda: self._config_registry.get_config(self.CONFIG_ENABLE_DYNAMIC_PROMPT),
            processor=_dynamic_process,
            process_name="DYNAMIC_PROMPT_INTERPRET",
        )

        def _translate_process(pos_prompt: str, neg_prompt: str) -> Tuple[str, str]:
            pos_prompt = translate("en", pos_prompt, "auto")
            neg_prompt = translate("en", neg_prompt, "auto")
            return pos_prompt, neg_prompt

        processor.register(
            judge=lambda: self._config_registry.get_config(self.CONFIG_ENABLE_TRANSLATE) and translate,
            processor=_translate_process,
            process_name="TRANSLATE",
        )

        def _shuffle_process(pos_prompt: str, neg_prompt: str) -> Tuple[str, str]:
            from random import shuffle

            pos_tokens = pos_prompt.split(",")
            shuffle(pos_tokens)
            pos_prompt: str = ",".join(pos_tokens)
            neg_tokens = neg_prompt.split(",")
            shuffle(neg_tokens)
            neg_prompt: str = ",".join(neg_tokens)
            return pos_prompt, neg_prompt

        processor.register(
            judge=lambda: self._config_registry.get_config(self.CONFIG_ENABLE_SHUFFLE_PROMPT),
            processor=_shuffle_process,
            process_name="SHUFFLE",
        )

        from graia.ariadne import Ariadne

        from graia.ariadne.model import Friend

        from graia.ariadne.event.message import FriendMessage

        @self.receiver(
            FriendMessage,
            decorators=[ContainKeyword(keyword=self._config_registry.get_config(self.CONFIG_POS_KEYWORD))],
        )
        @self.receiver(
            GroupMessage,
            decorators=[ContainKeyword(keyword=self._config_registry.get_config(self.CONFIG_POS_KEYWORD))],
        )
        async def diffusion(
            app: Ariadne,
            target: Union[Group, Friend],
            message: MessageChain,
            message_event: Union[GroupMessage, FriendMessage],
        ):
            """
            Asynchronously performs diffusion on the given message and sends the resulting image as a message in the group or to a friend.

            Args:
                app (Ariadne): The Ariadne instance.
                target (Union[Group, Friend]): The target group or friend to send the image to.
                message (MessageChain): The message containing the prompts for diffusion.
                message_event (Union[GroupMessage, FriendMessage]): The message event.

            Returns:
                None
            """
            # Extract positive and negative prompts from the message
            pos_prompt, neg_prompt, batch_count = extract_prompts(str(message), specify_batch_count=True)
            pos_prompt = ",".join(pos_prompt) if pos_prompt else get_default_pos_prompt()
            neg_prompt = ",".join(neg_prompt) if neg_prompt else get_default_neg_prompt()
            pos_prompt, neg_prompt = processor.process(pos_prompt, neg_prompt)
            # Create a diffusion parser with the prompts
            diffusion_paser = DiffusionParser(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                styles=self._config_registry.get_config(self.CONFIG_STYLES),
            )

            image_url = await _get_image_url(app, message_event)
            send_result = []
            for _ in range(batch_count):
                if image_url:
                    send_result.extend(await _make_img2img(diffusion_paser, image_url))

                else:
                    # Generate the image using the diffusion parser

                    send_result.extend(
                        await SD_app.txt2img(
                            diffusion_parameters=diffusion_paser,
                            HiRes_parameters=HiResParser(
                                enable_hr=self._config_registry.get_config(self.CONFIG_ENABLE_HR)
                            ),
                            output_dir=output_dir_path,
                        )
                    )
                if (
                    len(send_result) >= self.config_registry.get_config(self.CONFIG_SEND_BATCH_SIZE)
                    or _ + 1 == batch_count
                ):
                    # Send the image as a message in the group
                    await app.send_message(target, [Image(path=path) for path in send_result])
                    send_result.clear()

        @self.receiver(
            FriendMessage,
            decorators=[MatchRegex(regex=f"^{CMD.ROOT}\s+{CMD.CONTROLNET_CMD}\s+{CMD.CONTROLNET_DETECT_CMD}.*")],
        )
        @self.receiver(
            GroupMessage,
            decorators=[MatchRegex(regex=f"^{CMD.ROOT}\s+{CMD.CONTROLNET_CMD}\s+{CMD.CONTROLNET_DETECT_CMD}.*")],
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
                    img_to_base64(await download_file(save_dir=temp_dir_path, url=img.url)) for img in images
                ],
            )
            img_base64_list = await controlnet.detect(payload=pay_load)

            await app.send_message(target, [Image(base64=img_base64) for img_base64 in img_base64_list])

        from graia.ariadne.event.message import FriendMessage

        async def _get_image_url(app: Ariadne, message_event: Union[GroupMessage, FriendMessage]) -> str:
            """
            Retrieves the URL of an image from the given message event.

            Args:
                app (Ariadne): The Ariadne instance.
                message_event (Union[GroupMessage, FriendMessage]): The message event.

            Returns:
                str: The URL of the image, or None if no image is found.
            """
            image_url = ""
            if Image in message_event.message_chain:
                image_url = message_event.message_chain[Image, 1][0].url
            elif message_event.quote:
                target_to_query: List[int] = []
                target_to_query.append(message_event.quote.group_id) if message_event.quote.group_id else None
                target_to_query.extend([message_event.quote.sender_id, message_event.quote.target_id])

                for target in target_to_query:
                    origin_message: MessageChain = (
                        await app.get_message_from_id(message_event.quote.id, target=target)
                    ).message_chain
                    # check if the message contains a picture
                    image_url = origin_message[Image, 1][0].url if origin_message[Image, 1] else None
                    break
                    # FIXME cant get the quote message sent by the bot in the friend channel

            return image_url

        async def _make_img2img(diffusion_paser, image_url):
            # Download the first image in the chain
            print(f"Downloading image from: {image_url}\n")
            img_path = await download_file(save_dir=temp_dir_path, url=image_url)
            img_base64 = img_to_base64(img_path)
            cn_unit = None
            if self._config_registry.get_config(self.CONFIG_ENABLE_CONTROLNET):
                module: str = self._config_registry.get_config(self.CONFIG_CONTROLNET_MODULE)
                model: str = self._config_registry.get_config(self.CONFIG_CONTROLNET_MODEL)

                if module in controlnet.modules and model in controlnet.models:
                    cn_unit = ControlNetUnit(
                        input_image=img_base64,
                        module=module,
                        model=model,
                    )
            send_result = await SD_app.img2img(
                image_base64=img_base64,
                output_dir=output_dir_path,
                diffusion_parameters=diffusion_paser,
                controlnet_parameters=cn_unit,
            )
            return send_result
