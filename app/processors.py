from models import model_initializer
from torch.nn import functional as F
from video_llama.processors.video_processor import load_video


def process_video(video_path: str):
    model, vis_processor, _ = model_initializer.model, model_initializer.vis_processor, model_initializer.img_processor
    video = load_video(video_path=video_path, n_frms=16, height=224, width=224, sampling="uniform", return_msg=False)
    video = vis_processor.transform(video).unsqueeze(0).to(model_initializer.device)
    video_emb = model.encode_videoQformer_visual(video)[-1].last_hidden_state
    return F.normalize(model.vision_proj(video_emb), dim=-1)


def process_text(prompt: str):
    model = model_initializer.model
    inputs = model.tokenizer(prompt, padding="max_length", truncation=True, max_length=320, return_tensors="pt").to(
        model_initializer.device
    )
    embds = model.video_Qformer.bert(inputs.input_ids, inputs.attention_mask, return_dict=True).last_hidden_state
    return F.normalize(model.text_proj(embds[:, 0, :]), dim=-1)
