import torch
from typing import List, Iterator,Optional, Sequence, Sized, Union
from mmengine.dataset.sampler import InfiniteSampler
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class WeightedRandomSampler(InfiniteSampler):
    def __init__(self, 
                 weight_list: List,
                 replacement:bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert len(weight_list)==len(self.dataset.datasets)
        self.weight_list = weight_list
        self.replacement = replacement
        cummulative = self.dataset.cummulative_sizes
        weight = torch.as_tensor([weight_list[-1]]*cummulative[-1])
        for i in reversed(range(len(cummulative)-1)):
            weight[:cummulative[i]] = weight_list[i]
        self.weight = weight

    def _infinite_indices(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.multinomial(self.weight, self.size, self.replacement, generator=g).tolist()
            

