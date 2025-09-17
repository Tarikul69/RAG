from langchain_community.llms import LlamaCpp
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

llm = LlamaCpp(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # local model
    n_ctx=2048,
    n_threads=6
)


 

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)
