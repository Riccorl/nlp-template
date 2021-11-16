from pathlib import Path
from typing import Union, List, Any, Dict, Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, name: str, path: Union[str, Path, List[str], List[Path]], **kwargs):
        super().__init__()
        self.path = path
        self.name = name

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Dataset({self.name=}, {self.path=})"

    def load(self, paths: Union[str, Path, List[str], List[Path]]) -> Any:
        # load data from single or multiple paths in one single dataset
        raise NotImplementedError
