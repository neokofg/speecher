import time
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class TranslationModel:
    def __init__(self, model_id="haoranxu/ALMA-7B"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = self.model.to("cuda")

    def translate(self, text, source_lang, target_lang):
        prompt_template = PromptTemplate.from_template(
            f"You're a speech translator. Translate given text from {source_lang} to {target_lang}, don't add the original text, only output the translated text:\n{source_lang}: {{input_text}}"
        )

        prompt = prompt_template.format(input_text=text)
        # Tokenize the prompt and get the token length.
        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True)
        input_ids = encoding.input_ids.cuda()
        prompt_length = input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=3,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.7,
                top_p=0.8,
            )

        # Decode the full output and then remove the prompt part.
        full_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Alternatively, slice out the generated part using token ids.
        # Remove the tokens corresponding to the prompt.
        generated_tokens = generated_ids[0][prompt_length:]
        translation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return translation


def main():
    translator = TranslationModel()

    text = "Привет, это Егор и сегодня мы поговорим об механизме внимания"
    source_lang = "Russian"
    target_lang = "English"

    translation = translator.translate(text, source_lang, target_lang)

    return translation

if __name__ == "__main__":
    main()