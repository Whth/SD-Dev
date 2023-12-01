import random
from typing import List, Tuple, Any, Dict, Optional

from aiohttp import ClientSession
from pydantic import BaseModel, Field, PrivateAttr

from modules.shared import img_to_base64
from .api import API_GET_CONFIG

__DEFAULT_NEGATIVE_PROMPT__ = (
    "poorly drawn face,mutation,blurry,malformed limbs,disfigured,missing arms,missing legs,deformed legs,"
    "bad anatomy,bad hands,text,error,missing fingers,worst quality,normal quality,jpeg artifacts,signature,"
    "watermark,username,bad feet,poorly drawn asymmetric eyes,cloned face,mutilated,multiple breasts,"
    "poorly drawn hands,extra legs,malformed hands,long neck,three arms,long body,more than 2 thighs,"
    "more than 2 nipples,lowres,__low__"
)

__DEFAULT_POSITIVE_PROMPT__ = (
    "modern art,student uniform,white OR blue shirt,short blue skirt,white tights,"
    "high school girl,one girl,solo,upper body,shy,extremely cute,lovely,outside,"
    "on street,beautiful,expressionless,cool girl,medium breasts,watercolor,oil,"
    "see through,thighs,thin torso,masterpiece,wonderful art,high resolution,"
    "hair ornament,stripes,body curve,suitable for all viewers,"
    "__high__,__breasts__,__perspective__,__haircolor__,__hairstyle__,"
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

    batch_size: int = 1
    prompt: str = Field(default_factory=get_default_pos_prompt)
    negative_prompt: str = Field(default_factory=get_default_neg_prompt)
    styles: List[str] = Field(default_factory=list)
    seed: int = -1
    sampler_name: str = "UniPC"
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = Field(default_factory=get_shot_width)
    height: int = Field(default_factory=get_shot_height)


class RefinerParser(BaseModel):
    class Config:
        allow_mutation = False
        validate_assignment = True

    refiner_checkpoint: str = Field(default="")
    refiner_switch_at: float = Field(default=0.8, ge=0.0, le=1.0)


class HiResParser(BaseModel):
    """
    use to parse hires config_registry
    """

    enable_hr: bool = False
    denoising_strength: float = 0.57
    hr_scale: float = 1.56
    hr_upscaler: str = "Latent (antialiased)"
    hr_checkpoint_name: str = Field(default=None)
    hr_sampler_name: str = Field(default=None)
    hr_prompt: str = Field(default=None)
    hr_negative_prompt: str = Field(default=None)


class InterrogateParser(BaseModel):
    image_path: str = Field("", exclude=True)
    image: str = Field("")
    model: str = Field("deepdanbooru")

    def __init__(self, **data):
        super().__init__(**data)
        if self.image_path and self.image:
            raise ValueError("image_path and image cannot both be set.")
        elif self.image_path:
            self.image = img_to_base64(self.image_path)


class OverRideSettings(BaseModel):
    class Config:
        allow_mutation = False
        validate_assignment = True

    override_settings: Dict = Field(default_factory=dict)
    override_settings_restore_afterwards: bool = False


class Options(BaseModel):
    class Config:
        allow_mutation = True
        validate_assignment = True

    host_url: str = Field(exclude=True)
    samples_save: bool = True
    samples_format: str = "png"
    samples_filename_pattern: str = "[seed]"
    save_images_add_number: bool = True
    grid_save: bool = True
    grid_format: str = "png"
    grid_extended_filename: bool = False
    grid_only_if_multiple: bool = True
    grid_prevent_empty_spots: bool = False
    grid_zip_filename_pattern: str = ""
    n_rows: float = -1.0
    font: str = ""
    grid_text_active_color: str = "#000000"
    grid_text_inactive_color: str = "#999999"
    grid_background_color: str = "#ffffff"
    enable_pnginfo: bool = True
    save_txt: bool = False
    save_images_before_face_restoration: bool = False
    save_images_before_highres_fix: bool = False
    save_images_before_color_correction: bool = False
    save_mask: bool = False
    save_mask_composite: bool = False
    jpeg_quality: int = 80
    webp_lossless: bool = False
    export_for_4chan: bool = True
    img_downscale_threshold: int = 4
    target_side_length: int = 4000
    img_max_size_mp: int = 200
    use_original_name_batch: bool = True
    use_upscaler_name_as_suffix: bool = False
    save_selected_only: bool = True
    save_init_img: bool = False
    temp_dir: str = ""
    clean_temp_dir_at_start: bool = False
    save_incomplete_images: bool = False
    outdir_samples: str = ""
    outdir_txt2img_samples: str = "outputs/txt2img-images"
    outdir_img2img_samples: str = "outputs/img2img-images"
    outdir_extras_samples: str = "outputs/extras-images"
    outdir_grids: str = ""
    outdir_txt2img_grids: str = "outputs/txt2img-grids"
    outdir_img2img_grids: str = "outputs/img2img-grids"
    outdir_save: str = "log/images"
    outdir_init_images: str = "outputs/init-images"
    save_to_dirs: bool = True
    grid_save_to_dirs: bool = True
    use_save_to_dirs_for_ui: bool = False
    directories_filename_pattern: str = "[date]"
    directories_max_prompt_words: int = 8
    ESRGAN_tile: int = 192
    ESRGAN_tile_overlap: int = 8
    realesrgan_enabled_models: List = ["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"]
    upscaler_for_img2img: Optional[str] = None
    face_restoration: bool = False
    face_restoration_model: str = "CodeFormer"
    code_former_weight: float = 0.5
    face_restoration_unload: bool = False
    auto_launch_browser: str = "Local"
    show_warnings: bool = False
    show_gradio_deprecation_warnings: bool = True
    memmon_poll_rate: int = 8
    samples_log_stdout: bool = False
    multiple_tqdm: bool = True
    print_hypernet_extra: bool = False
    list_hidden_files: bool = True
    disable_mmap_load_safetensors: bool = False
    hide_ldm_prints: bool = True
    api_enable_requests: bool = True
    api_forbid_local_requests: bool = True
    api_useragent: str = ""
    unload_models_when_training: bool = False
    pin_memory: bool = False
    save_optimizer_state: bool = False
    save_training_settings_to_txt: bool = True
    dataset_filename_word_regex: str = ""
    dataset_filename_join_string: str = " "
    training_image_repeats_per_epoch: int = 1
    training_write_csv_every: int = 500
    training_xattention_optimizations: bool = False
    training_enable_tensorboard: bool = False
    training_tensorboard_save_images: bool = False
    training_tensorboard_flush_every: int = 120
    sd_model_checkpoint: str = "string"
    sd_checkpoints_limit: int = 1
    sd_checkpoints_keep_in_cpu: bool = True
    sd_checkpoint_cache: int = 0
    sd_unet: str = "Automatic"
    enable_quantization: bool = False
    enable_emphasis: bool = True
    enable_batch_seeds: bool = True
    comma_padding_backtrack: int = 20
    CLIP_stop_at_last_layers: int = 1
    upcast_attn: bool = False
    randn_source: str = "GPU"
    tiling: bool = False
    hires_fix_refiner_pass: str = "second pass"
    sdxl_crop_top: int = 0
    sdxl_crop_left: int = 0
    sdxl_refiner_low_aesthetic_score: float = 2.5
    sdxl_refiner_high_aesthetic_score: float = 6
    sd_vae_explanation: str = (
        "<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a "
        "standard <abbr title='red/green/blue'>RGB</abbr>\nimage into latent space "
        "representation and back. Latent space representation is what stable diffusion is "
        "working on during sampling\n(i.e. when the progress bar is between empty and full). "
        "For txt2img VAE is used to create a resulting image after the sampling is "
        "finished.\nFor img2img VAE is used to process user's input image before the sampling "
        "and to create an image after sampling."
    )
    sd_vae_checkpoint_cache: int = 0
    sd_vae: str = "Automatic"
    sd_vae_overrides_per_model_preferences: bool = True
    auto_vae_precision: bool = True
    sd_vae_encode_method: str = "Full"
    sd_vae_decode_method: str = "Full"
    inpainting_mask_weight: int = 1
    initial_noise_multiplier: int = 1
    img2img_extra_noise: int = 0
    img2img_color_correction: bool = False
    img2img_fix_steps: bool = False
    img2img_background_color: str = "#ffffff"
    img2img_editor_height: int = 720
    img2img_sketch_default_brush_color: str = "#ffffff"
    img2img_inpaint_mask_brush_color: str = "#ffffff"
    img2img_inpaint_sketch_default_brush_color: str = "#ffffff"
    return_mask: bool = False
    return_mask_composite: bool = False
    cross_attention_optimization: str = "Automatic"
    s_min_uncond: int = 0
    token_merging_ratio: int = 0
    token_merging_ratio_img2img: int = 0
    token_merging_ratio_hr: int = 0
    pad_cond_uncond: bool = False
    persistent_cond_cache: bool = True
    batch_cond_uncond: bool = True
    use_old_emphasis_implementation: bool = False
    use_old_karras_scheduler_sigmas: bool = False
    no_dpmpp_sde_batch_determinism: bool = False
    use_old_hires_fix_width_height: bool = False
    dont_fix_second_order_samplers_schedule: bool = False
    hires_fix_use_firstpass_conds: bool = False
    use_old_scheduling: bool = False
    interrogate_keep_models_in_memory: bool = False
    interrogate_return_ranks: bool = False
    interrogate_clip_num_beams: int = 1
    interrogate_clip_min_length: int = 24
    interrogate_clip_max_length: int = 48
    interrogate_clip_dict_limit: int = 1500
    interrogate_clip_skip_categories: List = []
    interrogate_deepbooru_score_threshold: float = 0.5
    deepbooru_sort_alpha: bool = True
    deepbooru_use_spaces: bool = True
    deepbooru_escape: bool = True
    deepbooru_filter_tags: str = ""
    extra_networks_show_hidden_directories: bool = True
    extra_networks_hidden_models: str = "When searched"
    extra_networks_default_multiplier: float = 1
    extra_networks_card_width: int = 0
    extra_networks_card_height: int = 0
    extra_networks_card_text_scale: float = 1
    extra_networks_card_show_desc: bool = True
    extra_networks_add_text_separator: str = " "
    ui_extra_networks_tab_reorder: str = ""
    textual_inversion_print_at_load: bool = False
    textual_inversion_add_hashes_to_infotext: bool = True
    sd_hypernetwork: str = "None"
    localization: str = "None"
    gradio_theme: str = "Default"
    gradio_themes_cache: bool = True
    gallery_height: str = ""
    return_grid: bool = True
    do_not_show_images: bool = False
    send_seed: bool = True
    send_size: bool = True
    js_modal_lightbox: bool = True
    js_modal_lightbox_initially_zoomed: bool = True
    js_modal_lightbox_gamepad: bool = False
    js_modal_lightbox_gamepad_repeat: int = 250
    show_progress_in_title: bool = True
    samplers_in_dropdown: bool = True
    dimensions_and_batch_together: bool = True
    keyedit_precision_attention: float = 0.1
    keyedit_precision_extra: float = 0.05
    keyedit_delimiters: str = ".\\/!?%^*;:{}=`~()"
    keyedit_move: bool = True
    quicksettings_list: List = ["sd_model_checkpoint"]
    ui_tab_order: List = []
    hidden_tabs: List = []
    ui_reorder_list: List = []
    hires_fix_show_sampler: bool = False
    hires_fix_show_prompts: bool = False
    disable_token_counters: bool = False
    add_model_hash_to_info: bool = True
    add_model_name_to_info: bool = True
    add_user_name_to_info: bool = False
    add_version_to_infotext: bool = True
    disable_weights_auto_swap: bool = True
    infotext_styles: str = "Apply if any"
    show_progressbar: bool = True
    live_previews_enable: bool = True
    live_previews_image_format: str = "png"
    show_progress_grid: bool = True
    show_progress_every_n_steps: bool = 10
    show_progress_type: str = "Approx NN"
    live_preview_allow_lowvram_full: bool = False
    live_preview_content: str = "Prompt"
    live_preview_refresh_period: bool = 1000
    live_preview_fast_interrupt: bool = False
    hide_samplers: List = []
    eta_ddim: float = 0
    eta_ancestral: float = 1
    ddim_discretize: str = "uniform"
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = 0
    s_noise: float = 1
    k_sched_type: str = "Automatic"
    sigma_min: float = 0
    sigma_max: float = 0
    rho: float = 0
    eta_noise_seed_delta: float = 0
    always_discard_next_to_last_sigma: bool = False
    sgm_noise_multiplier: bool = False
    uni_pc_variant: str = "bh1"
    uni_pc_skip_type: str = "time_uniform"
    uni_pc_order: float = 3
    uni_pc_lower_order_final: bool = True
    postprocessing_enable_in_main_ui: List = []
    postprocessing_operation_order: List = []
    upscaling_max_images_in_cache: bool = 5
    disabled_extensions: List = []
    disable_all_extensions: str = "none"
    restore_config_state_file: str = ""
    sd_checkpoint_hash: str = ""

    _fetched: bool = PrivateAttr(default=False)
    _hall_bak_dict: Dict[str, Any] = PrivateAttr(default={})
    _changed_bak_dict: Dict[str, Any] = PrivateAttr(default={})

    async def fetch_config(self, session: Optional[ClientSession] = None):
        """
        Fetches the configuration from the specified API endpoint.

        Raises:
            KeyError: If a key in the configuration dictionary is not found in the instance.

        Returns:
            None
        """
        if session:
            async with session.get(API_GET_CONFIG) as response:
                response = await response.json()
        else:
            async with ClientSession(base_url=self.host_url) as session:
                async with session.get(API_GET_CONFIG) as response:
                    response = await response.json()
        config_dict: Dict[str, Any] = response
        updated_config_count: int = self.load_dict_to_config(config_dict=config_dict)

        self._fetched = True

        print(f"Updated SD-config count: {updated_config_count}")

    def load_dict_to_config(self, config_dict: Dict[str, Any]):
        updated_config_count: int = 0
        for key, value in config_dict.items():
            if hasattr(self, key):
                if value != getattr(self, key):
                    setattr(self, key, value)
                    updated_config_count += 1
            else:
                raise KeyError(f'Key "{key}" not found in instance.')
        return updated_config_count

    def record_start(self):
        self._hall_bak_dict.clear()

        self._hall_bak_dict.update(self.dict())

    def record_end(self) -> Dict[str, Any]:
        """
        extract the diff between the start and end dictionaries.
        Returns:

        """
        temp = {}

        for key, value in self._hall_bak_dict.items():
            if value != getattr(self, key):
                temp[key] = getattr(self, key)  # store diff
                self._changed_bak_dict[key] = value  # store config been override

        return temp

    def generate_override_settings_payload(
        self, new_config: Dict[str, Any], recover_after_override: bool = True
    ) -> OverRideSettings:
        if all(hasattr(self, key) for key in new_config):
            self.revert_changes() if recover_after_override else None
            return OverRideSettings(
                override_settings=new_config, override_settings_restore_afterwards=recover_after_override
            )

        else:
            raise KeyError("illegal key found")

    def revert_changes(self):
        for key, value in self._changed_bak_dict.items():
            setattr(self, key, value)
        self._changed_bak_dict.clear()
