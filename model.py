from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from huggingface_hub import login
    
def initialize_llava():
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
    return llava_processor, llava_model

def initialize_mistral():
    login(token="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")
    mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
    mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    return mistral_model, mistral_tokenizer

