
from transformers import AutoProcessor, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
from PIL import Image
from dataclasses import dataclass
import base64
import io
import math
import json

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

# ------------------ DeepEyes Model ------------------
class DeepEyes:
    def __init__(self, model_path, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

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
        self.user_prompt = "
Think first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
        self.start_token = "<tool_call>"
        self.end_token = "</tool_call>"
        
        # Define stop tokens, including the end of tool call
        stop_words = [self.end_token, "<|im_end|>", "<|endoftext|>"]
        self.stop_words_ids = [self.tokenizer.encode(word)[0] for word in stop_words]


    def generate_output(self, messages):
        # History of the conversation
        chat_history = []
        # List of images in the conversation
        images = []

        # 1. Initial setup
        # Add system prompt
        chat_history.append({"role": "system", "content": self.system_prompt})
        
        # Add initial user message and image
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

        # 2. Multi-turn conversation loop
        while '</answer>' not in response_message and try_count < 10:
            
            # Prepare model inputs
            text = self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
            
            # Process images
            processed_images = []
            for img in images:
                ori_width, ori_height = img.size
                resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
                processed_images.append(img.resize((resize_w, resize_h), resample=Image.BICUBIC))

            inputs = self.processor(text=[text], images=processed_images, return_tensors="pt").to(self.device)

            # Generate response
            generated_ids = self.llm.generate(
                **inputs,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                do_sample=False if self.temperature == 0 else True,
                eos_token_id=self.stop_words_ids
            )
            
            # Decode and process the response
            generated_ids = generated_ids[0][len(inputs.input_ids[0]):]
            response_message = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            # Append assistant's raw response to history
            chat_history.append({"role": "assistant", "content": response_message})

            # 3. Handle tool calls
            if self.start_token in response_message and self.end_token in response_message:
                action_str = response_message.split(self.start_token)[1].split(self.end_token)[0].strip()
                try:
                    action = json.loads(action_str)
                    if action.get('name') == 'image_zoom_in_tool':
                        bbox = action['arguments']['bbox_2d']
                        left, top, right, bottom = map(int, bbox)
                        
                        # The tool operates on the last image added
                        source_image = images[-1]
                        cropped_image = source_image.crop((left, top, right, bottom))
                        images.append(cropped_image) # Add new image for the next turn
                        
                        # Create a tool response message pointing to the new image
                        tool_response_content = [
                            {"type": "text", "text": f"<tool_response><image_{len(images)}></tool_response>"}
                        ]
                        chat_history.append({"role": "user", "content": tool_response_content})

                except (json.JSONDecodeError, KeyError) as e:
                    # If tool call is malformed, add an error message and stop
                    chat_history.append({"role": "user", "content": "Error parsing tool call. Please provide the final answer."})
            
            try_count += 1

        # 4. Extract final answer
        if '</answer>' in response_message and '<answer>' in response_message:
            output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
        else: # Fallback if no answer tag is found
            # Clean up response by removing special tokens and tool calls
            final_text = response_message.split('<answer>')[-1]
            final_text = final_text.replace(self.start_token, "").replace(self.end_token, "")
            output_text = final_text.strip()

        return output_text

    def generate_outputs(self, messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res

# ------------------ Example Usage ------------------
if __name__ == "__main__":
    model_path = "Qwen/Qwen-VL-Chat" # Replace with your model path
    image_path = "path/to/your/image.jpg" # Replace with your image path

    model = DeepEyes(model_path, Args())
    img = Image.open(image_path).convert("RGB")

    messages = {"image": img, "prompt": "What is in this image?"}
    print(model.generate_output(messages))
