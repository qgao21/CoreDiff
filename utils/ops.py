from torch import nn
import torch
import os.path as osp
import os
from torchvision.utils import save_image
import torch.distributed as dist
import inspect

def turn_on_spectral_norm(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        if module.out_channels != 1 and module.in_channels > 4:
            module_output = nn.utils.spectral_norm(module)
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #     module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    else:
        world_size = 1
    if world_size is not None:
        rt /= world_size
    return rt


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict

def convert_to_ddp(*modules):
    modules = [x.cuda() for x in modules]
    if dist.is_initialized():
        rank = dist.get_rank()
        modules = [torch.nn.parallel.DistributedDataParallel(x,
                                                             device_ids=[rank, ],
                                                             output_device=rank) for x in modules]

    return modules