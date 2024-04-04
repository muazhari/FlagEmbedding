import torch
from transformers import is_torch_npu_available

from FlagEmbedding.visual import Visualized_BGE


class VisualizedBGE:
    def __init__(
            self,
            model_name_bge: str = None,
            model_weight: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            negatives_cross_device: bool = False,
            temperature: float = 0.02,
            use_fp16: bool = True,
            device: str = None
    ) -> None:
        self.model = Visualized_BGE(
            model_name_bge=model_name_bge,
            model_weight=model_weight,
            sentence_pooling_method=pooling_method,
            normlized=normalize_embeddings,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature
        )

        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        if device is None:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                print(f"----------using {self.num_gpus}*GPUs----------")
                self.model.model = torch.nn.DataParallel(self.model.model)
        else:
            self.num_gpus = 1

        self.model.eval()

    @torch.inference_mode()
    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)
