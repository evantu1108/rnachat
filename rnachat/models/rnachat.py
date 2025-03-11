import torch
from torch import nn
from rnachat.common.registry import registry
from rnachat.models.blip2 import Blip2Base, disabled_train
from RiNALMo.rinalmo.pretrained import get_pretrained_model
from rnachat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer,BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from argparse import ArgumentParser
from typing import List
@registry.register_model("rnachat")

# def get_device_map() -> str:
#     return 'cuda' if torch.cuda.is_available() else 'cpu'

# device = get_device_map()  # 'cpu'
class RNAChat(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "",
    }
    
    def __init__(self,
                 device=torch.device("cpu"),
                 freeze_rna_encoder=True,
                 llama_model="",
                 freeze_llama=True,
                 freeze_lp=False,
                 max_txt_len=32,
                 low_resource=False,  # use 8 bit and put vit in cpu
                 end_sym='\n',):
        super().__init__()
        print('Loading RNA encoder')
        self.rna_encoder, self.alphabet = get_pretrained_model(model_name="giga-v1")
        if freeze_rna_encoder:
            for name, param in self.rna_encoder.named_parameters():
                param.requires_grad = False
            self.rna_encoder = self.rna_encoder.eval()
            self.rna_encoder.train = disabled_train
            # logging.info("freeze rna encoder")
        else:
            self.rna_encoder = self.rna_encoder.train()
            
        parser = ArgumentParser()
        self.args_ = parser.parse_args()
        self.args_.device = torch.cuda.current_device()
        self.low_resource = low_resource
        
        self.tokenizer = self.alphabet.batch_tokenize
        print('Loading LLAMA model')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        if self.low_resource:
            print("Start Low Resource Mode")
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16
            #     )
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto',
                # quantization_config=bnb_config
                # load_in_8bit_fp32_cpu_offload=True,
                # device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                device_map='auto',
            )

        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules: List[str] = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        self.rinalmo_llama_proj = nn.Linear(
            1280, self.llama_model.config.hidden_size
        )

        if freeze_lp:
            for name, param in self.rinalmo_llama_proj.named_parameters():
                param.requires_grad = False
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
       

    def encode_rna(self, rna):
        inputs = torch.tensor(self.alphabet.batch_tokenize(rna), dtype=torch.int64, device=self.device).to(torch.cuda.current_device())
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.rna_encoder(inputs)['representation']
        if outputs.dtype != self.rinalmo_llama_proj.weight.dtype:
            outputs = outputs.to(self.rinalmo_llama_proj.weight.dtype)

        inputs_llama = self.rinalmo_llama_proj(outputs)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(outputs.device)
        return inputs_llama, atts_llama

    def prompt_list_wrap(self, rna_embeds, atts_rna, prompt):
        if prompt:
            p_before_lst = []
            p_after_lst = []
            for p in prompt:
                p_before, p_after = p.split('<RNAHere>')
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)
            p_before_tokens_lst = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False).to(rna_embeds.device)

            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(rna_embeds.device)
            
            p_before_embeds = self.llama_model.get_input_embeddings()(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.get_input_embeddings()(p_after_tokens_lst.input_ids)
            wrapped_rna_embeds = torch.cat([p_before_embeds, rna_embeds, p_after_embeds], dim=1)
            wrapped_atts_rna = atts_rna[:, :1].expand(-1, wrapped_rna_embeds.shape[1])
            return wrapped_rna_embeds, wrapped_atts_rna
        else:
            return rna_embeds, atts_rna

    def forward(self, samples):
        seqs = samples["seq"] # list of seq
        # print(samples)
        rna_embeds, atts = self.encode_rna(seqs)

        rna_embeds, atts_rna = self.prompt_list_wrap(rna_embeds, atts, samples["prompt"])

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(rna_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_rna.shape[0], atts_rna.shape[1]+1],
                       dtype=torch.long).to(rna_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = rna_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.get_input_embeddings()(bos)
        atts_bos = atts_rna[:, :1]

        to_regress_embeds = self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, rna_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_rna, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        logits = outputs.logits
        logits = torch.argmax(logits, dim=2)
        loss = outputs.loss
        return {"loss": loss}
    
    @classmethod
    def from_config(cls, cfg):

        llama_model = cfg.get("llama_model")

        freeze_rna_encoder = cfg.get("freeze_rna_encoder", False)
        freeze_lp = cfg.get("freeze_lp", False)
        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)

        model = cls(
            device= device_8bit,
            freeze_rna_encoder=freeze_rna_encoder,
            freeze_lp=freeze_lp,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            # embedding_agg = embedding_agg, 
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            # device_8bit=device_8bit,
        )

        stage1_ckpt = cfg.get("stage1_ckpt", "")  # load weights of encoder and LP
        if stage1_ckpt:
            import os
            print(os.getcwd())
            print("Load GLM and LP Checkpoint: {}".format(stage1_ckpt))
            ckpt = torch.load(stage1_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        
        peft_ckpt = cfg.get("peft_ckpt", "")  # load weights of LoRA
        if peft_ckpt:
            print("Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            
        return model