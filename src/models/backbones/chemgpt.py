import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.backbones.base import MoleculeEncoder


@MoleculeEncoder.register("chemgpt")
class ChemGPTEncoder(MoleculeEncoder):
    def __init__(self):
        model_name = "ncfrey/ChemGPT-1.2B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def encode(self, smiles: str) -> np.ndarray:
        tokens = self.tokenizer(smiles, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            embedding = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().flatten()

    def encode_batch(self, smiles_list: list) -> np.ndarray:
        tokens = self.tokenizer(smiles_list, return_tensors="pt", padding=True).to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            embedding = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()
