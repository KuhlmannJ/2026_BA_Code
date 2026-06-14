import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Thinking"
# MODEL_NAME = "Qwen/Qwen3-VL-32B-Thinking"         # 66.7GB VRAM
# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking"

#### Helping Functions ##########################################
# Some segmentation for log readablility
def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

#### 1. GPU Details #############################################
banner("STEP 1: GPU / CUDA")
props      = torch.cuda.get_device_properties(0)
gpu_name   = torch.cuda.get_device_name(0)
vram_total = props.total_memory / 1e9
gpu_uuid   = props.uuid
print(f"  GPU  : {gpu_name}")
print(f"  VRAM : {vram_total:.1f} GB")
print(f"  UUID : {gpu_uuid}")


#### 2. VLM Loading #############################################
banner("STEP 2: LOAD VLM")

match MODEL_NAME:
    case "Qwen/Qwen3-VL-235B-A22B-Thinking":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0',
        )
    case "Qwen/Qwen3-VL-32B-Thinking":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0',
        )
    case "Qwen/Qwen3-VL-30B-A3B-Thinking":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda:0',
        )

print(f" Loaded: {MODEL_NAME}")
print(f" Attention loaded:{model.config._attn_implementation}")
print(f" VRAM belegt: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)