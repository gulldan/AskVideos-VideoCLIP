from argparse import Namespace

from config import settings

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.processors import Blip2ImageEvalProcessor


class ModelInitializer:
    def __init__(self) -> None:
        self.device = settings.DEVICE
        self.eval_config = settings.EVAL_CONFIG
        self.gpu_id = int(settings.GPU_ID)
        self.model, self.vis_processor, self.img_processor = self.init_model()

    def init_model(self):
        args = {"cfg_path": self.eval_config, "gpu_id": self.gpu_id, "options": None}
        args = Namespace(**args)
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = self.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(f"cuda:{self.gpu_id}")
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        img_processor = Blip2ImageEvalProcessor()
        return model, vis_processor, img_processor


model_initializer = ModelInitializer()
