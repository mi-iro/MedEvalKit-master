import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import transformers
import tokenizers

from PIL import Image
from tqdm import tqdm
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


from .llava import conversation as conversation_lib
from .llava.model import *
from .llava.mm_utils import tokenizer_image_token,process_images
from .llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig

from .utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square,com_vision_args


class HealthGPT:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LlavaQwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_path,
        attn_implementation="flash_attention_2",
        torch_dtype= torch.float16,
        device_map="cuda"
    )
        from .llava.peft import LoraConfig, get_peft_model
        print("load model done")
        lora_config = LoraConfig(
            r= 32,
            lora_alpha=64,
            target_modules=find_all_linear_names(self.llm),
            lora_dropout=0.0,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=4,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        print("load lora done")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=False,
        )
        if self.tokenizer.unk_token:
            self.tokenizer.pad_token = tokenizer.unk_token
        else:
            self.tokenizer.legacy = False
        print("load tokenizer done")

        from .utils import com_vision_args
        com_vision_args.model_name_or_path = model_path
        com_vision_args.vision_tower = '/mnt/workspace/workgroup_dev/longli/models/hub/clip-vit-large-patch14-336'
        com_vision_args.version = "qwen_2"

        self.llm.get_model().initialize_vision_modules(model_args=com_vision_args)
        self.llm.get_vision_tower().to(dtype=torch.float16)
        self.llm.get_model().mm_projector.to(dtype=torch.float16)
        print("load vision tower done")

        self.llm = load_weights(self.llm, "/mnt/workspace/workgroup_dev/longli/models/hub/HealthGPT-XL32/com_hlora_weights_QWEN_32B.bin")
        print("load weights done")
        self.llm.eval()
        self.llm.to(dtype=torch.float16).cuda()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens


    def process_messages(self,messages):
        conv = conversation_lib.conv_templates["qwen_2"].copy()
        conv.messages = []
        if  "system" in messages:
            conv.system = messages["system"]
        
        imgs = []
        if "image" in messages:
            image = messages["image"]
            if isinstance(image,str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image,Image.Image):
                image = image.convert('RGB')
            
            
            imgs.append(image)
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + messages["prompt"]
        elif "images" in messages:
            images = messages["images"]
            prompt = ""
            for i,image in enumerate(images):
                prompt += f"<image_{i+1}>: " + DEFAULT_IMAGE_TOKEN + '\n'
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                elif isinstance(image,Image.Image):
                    image = image.convert("RGB")
                imgs.append(image)
            prompt += messages["prompt"]
        else:
            prompt = messages["prompt"]
        conv.append_message(conv.roles[0],prompt)
        conv.append_message(conv.roles[1],None) 
        prompt = conv.get_prompt()
        imgs = None if len(imgs) == 0 else imgs
        return prompt,imgs


    def generate_output(self,messages):
        prompt,imgs = self.process_messages(messages)
        if imgs:
            # imgs = imgs[0]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze_(0).cuda()
            imgs = [expand2square(img, tuple(int(x*255) for x in self.llm.get_vision_tower().image_processor.image_mean)) for img in imgs]
            imgs = self.llm.get_vision_tower().image_processor.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.float16, device='cuda', non_blocking=True)
            # imgs = process_images(imgs,self.llm.get_vision_tower().image_processor,self.llm.config)
        else:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze_(0).cuda()
            imgs = None

        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            output_ids = self.llm.base_model.model.generate(input_ids,images=imgs,do_sample=do_sample,num_beams=5,max_new_tokens=self.max_tokens,temperature = self.temperature,top_p = self.top_p,repetition_penalty = self.repetition_penalty,use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_outputs(self,messages_list):
        outputs = []
        for messages in tqdm(messages_list):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs

if __name__ == '__main__':
    # def construct_messages(sample):
    #     question = sample["question"]
    #     image = sample["image"]
    #     answer = sample["answer"]
    #     if answer in ["yes","no"]:
    #         question += "Please output 'yes' or 'no'(no extra output)"
    #         if os.environ.get("STRICT_PROMPT","False") == "True":
    #             question += "Wrap your final answer between <answer> and </answer>, for example: <answer>yes</answer>"
    #     else:
    #         question += 'Answer the question concisely(Keep it short and concise).'
    #         if os.environ.get("STRICT_PROMPT","False") == "True":
    #             question += "Wrap your final answer between <answer> and </answer>, for example: <answer>positive influence</answer>"
    #     messages = {"prompt":question,"image":image}
    #     sample["messages"] = messages
    #     return sample
    
    # from datasets import load_dataset
    # from tqdm import tqdm
    # dataset = load_dataset("/mnt/workspace/workgroup_dev/longli/MedLLMBenchmarks/benchmarks/VQA_RAD",split="test")

    # llm = HealthGPT()
    # for data in tqdm(dataset):
    #     data = construct_messages(data)
    #     image = data["image"]
    #     messages = {"prompt":data["question"],"image":image}
    #     result = llm.generate_output(messages)

    llm = HealthGPT()
    image = "/mnt/workspace/workgroup_dev/longli/MedLLMBenchmarks/benchmarks/OmniMedVQA/Images/ACRIMA/Im553_g_ACRIMA.png"
    image2 = "/mnt/workspace/workgroup_dev/longli/MedLLMBenchmarks/benchmarks/OmniMedVQA/Images/BreakHis/benign/SOB_B_A-14-22549AB-40-005.png"
    messages = [{ 'images': [image,image2],"prompt":"\nQuestion:  你是一个医疗助手，我的问题是这两张图片里的内容是什么？"}]
    result = llm.generate_outputs(messages)
    print("result:",result)