import os
import sys
from proteinchat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random
import pandas as pd
questions = ["Tell me about this protein.", 
                "What is the functionality of this protein?", 
                "Briefly summarize the functionality of this protein.",
                "Please provide a detailed description of the protein."]
q_map = {
    "Can this protein bind to RNA?":
    " Reply only with Yes or No.",
    "Can this protein bind to DNA?":
    " Reply only with Yes or No.",
    "What type of enzyme is this?":
    " Choose one from Transferase, Hydrolase, Oxidoreductase, Ligase, Lyase, Isomerase, and Translocase.",
    "What type of protein is this?":
    " Choose one from Ribonucleoprotein and Chaperone protein",
    "What electron acceptor or cofactor does this enzyme use?":
    " Choose one from NAD and NADP.",
    "What ligand can this protein bind to?":
    " Choose one from Nucleotide, Magnesium, Zinc, Iron, S-adenosyl-L-methionine, and Manganese.",
    "Which cellular or extracellular component can this protein be found in?":
    " Choose one from Cytoplasm, Membrane, Nucleus, Secreted, Mitochondrion, and Plastid",
    "What biological process does this protein involved in?":
    " Choose one from Molecule Transport, Transcription from DNA to mRNA, Amino-acid biosynthesis, Protein biosynthesis from mRNA molecules, Lipid metabolism, tRNA processing, DNA damage, and Cell cycle."
}
class RNADataset(BaseDataset):
    def __init__(self, seq_path):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # print("______Enter Seq Dataset____")
        # super().__init__(vis_processor, text_processor)
        # self.qa_path = qa_path
        # self.seq_path = seq_path

        df = pd.read_csv("rna_summary_2d.csv")
        self.sequence = df['Sequence'].values.tolist()[:4200]
        self.labels = df['summary_no_citation'].values.tolist()[:4200]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        
        # if index < self.split1: # sample kw 
        #     uniprot_id = self.kw[index]["uniprot_id"]
        #     answer = self.kw[index]["A"]
        #     query = self.kw[index]['Q']
        #     query += q_map[query]
        #     prompt = f"###Human: <protein><proteinHere></protein> {query} ###Assistant:"
        # elif index < self.split2: # sample rule based functionality
        #     true_index  = (index - self.split1) % self.len_rule
        #     uniprot_id = self.rule[true_index]["uniprot_id"]
        #     answer = self.rule[true_index]["caption"]
        #     prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
        # else: # sample manual annotated functionality
        #     true_index  = (index - self.split2) % self.len_manual
        #     uniprot_id = self.manual[true_index]["uniprot_id"]
        #     answer = self.manual[true_index]["caption"]
        #     prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
        
        seq = self.sequence[index]
        prompt =  f"###Human: <RNA><RNAHere></RNA> Give a functional description of this RNA. ###Assistant:"
        if len(seq) > 1000:
            seq = seq[:1000]
        return {
            "seq": seq,
            "text_input": self.labels[index],
            "prompt": prompt
        }




