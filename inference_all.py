import argparse
import os
import random
import time
import math
######## HF CACHE (LOAD BEFORE HF PACKAGES) ########
# os.environ['HF_HOME'] = "/data1/mingjia/cache/huggingface"
# print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from rnachat.common.config import Config
from rnachat.common.registry import registry
from rnachat.common.dist_utils import get_rank, init_distributed_mode
from rnachat.common.conversation import Chat, CONV_VISION

from eval import get_simcse, get_simcse_llm_param
import json

# imports modules for registration
from rnachat.datasets.builders import *
from rnachat.models import *
from rnachat.runners import *
from rnachat.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.",
                        default='configs/rnachat_eval.yaml')
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
init_distributed_mode(cfg.run_cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_rna(seq):
    chat_state = CONV_VISION.copy()
    img_list = []
    rna_emb, llm_message = chat.upload_rna(seq, chat_state, img_list)
    return chat_state, img_list, rna_emb

def gradio_ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state

def gradio_answer(chat_state, img_list, num_beams=1, temperature=1e-3, top_p=0.9, save_embeds=False):
    # print(chat_state)
    print(chat_state)
    llm_message, _, loss = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              top_p = top_p,
                              #repetition_penalty=2.0,
                              max_new_tokens=200,
                              max_length=1500, 
                              save_embeds=save_embeds)
    return llm_message, chat_state, img_list, loss


if  __name__ == "__main__":
    directory_name = "results"
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
        except Exception as e:
            print(f"An error occurred when creating results folder: {e}")

    df = pd.read_csv("rna_summary_2d.csv")
    ids = df['id'].values.tolist()[4200:4210]
    names = df['name'].values.tolist()[4200:4210]
    sequence = df['Sequence'].values.tolist()[4200:4210]
    labels = df['summary_no_citation'].values.tolist()[4200:4210]
    func_text = []
    loss_list = []
    for i, (id, name, seq, lab) in enumerate(zip(ids,names, sequence, labels)):

        if len(seq) > 1000:
            seq = seq[:1000]

        user_message = f"###Human: Give me a functional description of this RNA named {name}. ###Assistant:"
        chat_state, img_list, rna_embs = upload_rna(seq)
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=4, temperature=0.7)
    
        loss_list.append(loss)
        entry = {"seq": seq, "query": user_message, "correct_func": lab, "predict_func": llm_message}
        func_text.append(entry)
    
        print("Uniprot ID:", id)
        print("Correct summary:", lab)
        print(f"Predicted summary: {llm_message}")
        print('='*80)
    print("******************")
    simcse_path = "princeton-nlp/sup-simcse-roberta-large"
    scores = get_simcse(simcse_path, func_text)
    with open("results/rna.json", "a") as outfile:
        json.dump(scores, outfile, indent=4)


