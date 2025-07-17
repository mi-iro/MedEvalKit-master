from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration
import torch

class Qwen2_5_VL:
    def __init__(self, model_path, args):
        super().__init__()
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=None,            # 强制整张卡
            attn_implementation="flash_attention_2"
        ).to("cuda")                   # ✅ 确保模型在 GPU

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    # --------------------------------------------------
    def process_messages(self, messages):
        # 1. 构造对话
        new_messages = []
        if "system" in messages:
            new_messages.append({"role": "system", "content": messages["system"]})
        if "image" in messages:
            new_messages.append({"role": "user", "content": [
                {"type": "image", "image": messages["image"]},
                {"type": "text", "text": messages["prompt"]}
            ]})
        elif "images" in messages:
            content = []
            for i, img in enumerate(messages["images"]):
                content.append({"type": "text", "text": f"<image_{i+1}>: "})
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": messages["prompt"]})
            new_messages.append({"role": "user", "content": content})
        else:
            new_messages.append({"role": "user", "content": [
                {"type": "text", "text": messages["prompt"]}
            ]})

        prompt = self.processor.apply_chat_template(
            new_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2. 提取视觉数据
        image_inputs, video_inputs = process_vision_info(new_messages)

        # 3. 构建张量
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 4. ✅ 统一搬到 GPU
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        return inputs

    # --------------------------------------------------
    def generate_output(self, messages):
        inputs = self.process_messages(messages)   # ✅ 已全在 GPU
        do_sample = self.temperature != 0
        generated_ids = self.llm.generate(
            **inputs,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample
        )

        # 截取新生成的部分
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0]

    # --------------------------------------------------
    def generate_outputs(self, messages_list):
        res = []
        for messages in messages_list:
            res.append(self.generate_output(messages))
        return res