import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"

def load_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "torch_dtype": torch_dtype
    }

global_model = None
if __name__ == "__main__":
    global_model = load_model()
    print("✨ Модель успешно загружена и инициализирована 🥳 ✨")