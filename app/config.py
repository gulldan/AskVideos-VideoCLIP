import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEVICE: str = os.getenv("DEVICE", "cuda")
    EVAL_CONFIG: str = os.getenv("EVAL_CONFIG", "eval_configs/video_clip_v0.2.yaml")
    GPU_ID: str = os.getenv("GPU_ID", "0")
    HUGGINGFACE_HUB_CACHE: str = os.getenv("HUGGINGFACE_HUB_CACHE", "./models/hf")
    TORCH_HOME: str = os.getenv("TORCH_HOME", "./models/torch")


settings = Settings()
