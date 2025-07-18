from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from PIL import Image
from dataclasses import dataclass
import base64
import io
import math
import json
import os

# ------------------ Args ------------------
@dataclass
class Args:
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 8192

# ------------------ Image Utils ------------------
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def encode_pil_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# ------------------ DeepEyes Model (VLLM Implementation) ------------------
class DeepEyes:
    def __init__(self, model_path, args):
        super().__init__()
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=int(os.environ.get("tensor_parallel_size", 4)),
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": int(os.environ.get("max_image_num", 10))},
            mm_processor_kwargs={"use_fast": True},
            # enforce_eager=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.tokenizer = self.llm.get_tokenizer()


        self.system_prompt = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>"""
        self.user_prompt = "Think first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
        self.start_token = "<tool_call>"
        self.end_token = "</tool_call>"
        
        stop_words = [self.end_token, "<|im_end|>", "<|endoftext|>"]
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_new_tokens,
            stop=stop_words,
        )

    def process_messages(self, messages):
        chat_history = []
        images = []

        # 1. Initial setup
        chat_history.append({"role": "system", "content": self.system_prompt})

        pil_img = messages.get("image")
        if pil_img:
            images.append(pil_img)
            user_content = [
                {"type": "image"},
                {"type": "text", "text": messages["prompt"] + self.user_prompt}
            ]
        else:
            user_content = [{"type": "text", "text": messages["prompt"] + self.user_prompt}]
        chat_history.append({"role": "user", "content": user_content})
        
        response_message = ""
        try_count = 0
        
        while '</answer>' not in response_message and try_count < 10:
            text = self.processor.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
            
            processed_images = []
            for img in images:
                ori_width, ori_height = img.size
                resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
                processed_images.append(img.resize((resize_w, resize_h), resample=Image.BICUBIC))

            llm_inputs = {
                "prompt": text,
                "multi_modal_data": {"image": processed_images}
            }
            
            outputs = self.llm.generate([llm_inputs], self.sampling_params)
            response_message = outputs[0].outputs[0].text
            
            chat_history.append({"role": "assistant", "content": response_message})

            if self.start_token in response_message and self.end_token in response_message:
                action_str = response_message.split(self.start_token)[1].split(self.end_token)[0].strip()
                try:
                    action = json.loads(action_str)
                    if action.get('name') == 'image_zoom_in_tool':
                        bbox = action['arguments']['bbox_2d']
                        left, top, right, bottom = map(int, bbox)
                        
                        source_image = images[-1]
                        cropped_image = source_image.crop((left, top, right, bottom))
                        images.append(cropped_image)
                        
                        tool_response_content = [
                            {"type": "text", "text": "<tool_response>"},
                            {"type": "image"}, 
                            {"type": "text", "text": "</tool_response>"}
                        ]
                        chat_history.append({"role": "user", "content": tool_response_content})

                except (json.JSONDecodeError, KeyError):
                    chat_history.append({"role": "user", "content": "Error parsing tool call."})
            
            try_count += 1

        if '</answer>' in response_message and '<answer>' in response_message:
            output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
        else:
            final_text = response_message.split('<answer>')[-1]
            final_text = final_text.replace(self.start_token, "").replace(self.end_token, "")
            output_text = final_text.strip()
            
        return output_text

    def generate_output(self, messages):
        return self.process_messages(messages)

    def generate_outputs(self, messages_list):
        llm_inputs_list = []
        prompts = []
        multi_modal_data_list = []

        for messages in messages_list:
            chat_history = []
            images = []
            chat_history.append({"role": "system", "content": self.system_prompt})

            pil_img = messages.get("image")
            if pil_img:
                images.append(pil_img)
                user_content = [
                    {"type": "image"},
                    {"type": "text", "text": messages["prompt"] + self.user_prompt}
                ]
            else:
                user_content = [{"type": "text", "text": messages["prompt"] + self.user_prompt}]
            chat_history.append({"role": "user", "content": user_content})

            text = self.processor.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

            processed_images = []
            for img in images:
                ori_width, ori_height = img.size
                resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
                processed_images.append(img.resize((resize_w, resize_h), resample=Image.BICUBIC))
            
            multi_modal_data_list.append({"image": processed_images})

        # This is a simplified way to create llm_inputs for batching.
        # In a real-world scenario, you might need more sophisticated logic to handle different numbers of images per prompt.
        llm_inputs_list = [{"prompt": p, "multi_modal_data": m} for p, m in zip(prompts, multi_modal_data_list)]

        outputs = self.llm.generate(llm_inputs_list, self.sampling_params)
        
        res = []
        for output in outputs:
            response_message = output.outputs[0].text
            if '</answer>' in response_message and '<answer>' in response_message:
                output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
            else:
                final_text = response_message.split('<answer>')[-1]
                final_text = final_text.replace(self.start_token, "").replace(self.end_token, "")
                output_text = final_text.strip()
            res.append(output_text)
        return res

# ------------------ Example Usage ------------------
if __name__ == "__main__":
    model_path = "/mnt/b_public/data/libo/model/deepeyes-7B" 
    image_path = "/mnt/b_public/data/libo/MedEvalKit-master/pku.jpg"

    try:
        model = DeepEyes(model_path, Args())
        img = Image.open(image_path).convert("RGB")

        messages = {"image": img, "prompt": "What is in this image? Please describe it."}
        print(model.generate_output(messages))
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your model_path and image_path are correct.")
