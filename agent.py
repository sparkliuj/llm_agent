
import torch
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain import PromptTemplate

load_dotenv()
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map='auto',
    torch_dtype=torch.float32,
    quantization_config=quantization_config
)
pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)
pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
local_llm = HuggingFacePipeline(pipeline=pipe)

template = '''def print_prime(n):
   """
   Print all primes between 1 and 
   """'''
prompt = PromptTemplate(template=template, input_variables=[])
chain = prompt | local_llm
re = chain.invoke({})
print(re)
