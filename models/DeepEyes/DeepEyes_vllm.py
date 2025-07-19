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
        
        stop_words = [self.end_token, "<|im_end|>", "<|endoftext|>", '</answer>']
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_new_tokens,
            stop=stop_words,
        )

    def generate_outputs(self, messages_list):
        """
        以有状态、迭代的方式高效处理批量请求，并支持工具调用。
        """
        # 1. 初始化每个请求的状态
        # 使用一个列表来追踪每个请求的独立状态
        request_states = []
        for i, messages in enumerate(messages_list):
            pil_img = messages.get("image")
            images = [pil_img] if pil_img else []
            
            chat_history = [
                {"role": "system", "content": self.system_prompt}
            ]
            user_content = [
                {"type": "image"} if pil_img else None,
                {"type": "text", "text": messages["prompt"] + ' ' + self.user_prompt}
            ]
            user_content = [item for item in user_content if item is not None]
            chat_history.append({"role": "user", "content": user_content})

            request_states.append({
                "id": i,
                "status": "active",  # active, completed, failed
                "images": images,
                "chat_history": chat_history,
                "result": None,
                "try_count": 0
            })

        final_results = [None] * len(messages_list)
        max_iterations = 10 # 设置最大迭代次数，防止无限循环

        # 2. 主循环，直到所有活动请求都处理完毕
        for i in range(max_iterations):
            active_requests = [req for req in request_states if req["status"] == "active"]
            if not active_requests:
                break # 所有请求都已完成或失败

            # 3. 准备当前批次的输入
            llm_inputs_list = []
            for req in active_requests:
                # 准备提示
                text = self.processor.apply_chat_template(
                    req["chat_history"], tokenize=False, add_generation_prompt=True
                )
                
                # 准备图片
                processed_images = []
                for img in req["images"]:
                    ori_width, ori_height = img.size
                    resize_w, resize_h = smart_resize(ori_width, ori_height)
                    processed_images.append(img.resize((resize_w, resize_h), resample=Image.BICUBIC))

                llm_inputs_list.append({
                    "prompt": text,
                    "multi_modal_data": {"image": processed_images}
                })

            # 4. 调用 vLLM 进行批量推理
            outputs = self.llm.generate(llm_inputs_list, self.sampling_params)

            # 5. 分派结果并更新状态
            for idx, req_state in enumerate(active_requests):
                response_message = outputs[idx].outputs[0].text

                # 更新对话历史
                req_state["chat_history"].append({"role": "assistant", "content": response_message})
                req_state["try_count"] += 1

                # 检查是需要工具调用还是已完成
                # if self.start_token in response_message and self.end_token in response_message:
                if self.start_token in response_message:
                    # 情况A: 需要工具调用
                    try:
                        # 补充 </tool_call>
                        response_message += self.end_token
                        req_state["chat_history"][-1] = {"role": "assistant", "content": response_message}

                        action_str = response_message.split(self.start_token)[1].split(self.end_token)[0].strip()
                        action = json.loads(action_str)
                        if action.get('name') == 'image_zoom_in_tool':
                            bbox = action['arguments']['bbox_2d']
                            left, top, right, bottom = map(int, bbox)
                            
                            # 注意：这里需要明确裁剪哪张图，通常是第一张或最后一张
                            source_image = req_state["images"][0] 
                            cropped_image = source_image.crop((left, top, right, bottom))
                            req_state["images"].append(cropped_image)
                            
                            # 为下一轮准备工具响应
                            tool_response_content = [
                                {"type": "text", "text": "<tool_response>"},
                                {"type": "image"}, 
                                {"type": "text", "text": "</tool_response>"}
                            ]
                            req_state["chat_history"].append({"role": "user", "content": tool_response_content})
                        else:
                            req_state["status"] = "failed" # 未知的工具
                            req_state["result"] = "Error: Unknown tool call."

                    except Exception as e:
                        req_state["status"] = "failed"
                        req_state["result"] = f"Error parsing tool call: {e}"

                elif '<answer>' in response_message:
                    # 情况B: 已完成
                    req_state["status"] = "completed"
                    req_state["result"] = response_message.split('<answer>')[1].strip()
                elif req_state["try_count"] >= max_iterations:
                    # 情况C: 超时/失败
                    req_state["status"] = "failed"
                    req_state["result"] = "Error: Max iterations reached without a final answer."

        # 6. 整理并返回最终结果
        for req in request_states:
            final_results[req["id"]] = req["result"]
            
        return final_results

# ------------------ Example Usage ------------------
if __name__ == "__main__":
    # --- 1. 初始化模型和参数 ---
    model_path = "/mnt/b_public/data/libo/model/deepeyes-7B" 
    # 假设你有两张不同的图片用于测试
    image_path_1 = "/mnt/b_public/data/libo/MedEvalKit-master/40.jpg" # 包含红色箱子的图片
    image_path_2 = "/mnt/b_public/data/libo/MedEvalKit-master/pku.jpg"   # 另一张测试图片

    try:
        model = DeepEyes(model_path, Args())
        
        # --- 2. 准备图片资源 ---
        img1 = Image.open(image_path_1).convert("RGB")
        img2 = Image.open(image_path_2).convert("RGB")

        # --- 3. 构建批处理请求列表 (messages_list) ---
        # 这是一个包含多个请求字典的列表
        messages_list = [
            # 请求 1: 复杂查询，很可能需要调用 image_zoom_in_tool
            {
                "image": img1, 
                "prompt": "what is on red bin?"
            },
            
            # 请求 2: 简单的图片描述，可能不需要工具
            {
                "image": img2,
                "prompt": "please describe this picture."
            },
        ]
        
        print(f"准备发送 {len(messages_list)} 个请求到批处理引擎...")
        print("-" * 30)

        # --- 4. 调用批处理函数 ---
        # generate_outputs 函数会处理整个列表
        batch_results = model.generate_outputs(messages_list)

        # --- 5. 打印批处理结果 ---
        print("\n" + "="*15 + " 批处理完成 " + "="*15)
        for i, result in enumerate(batch_results):
            original_prompt = messages_list[i]["prompt"]
            print(f"\n--- 结果来自请求 {i+1} ---")
            print(f"原始提问: {original_prompt}")
            print(f"模型回答: {result}")
        print("\n" + "="*40)

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        print("请确保你的 model_path 和 image_path 路径是正确的。")
    except Exception as e:
        print(f"发生了一个未知错误: {e}")