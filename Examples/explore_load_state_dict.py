import torch
import os
import easyocr
import logging


def example_copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def example_torch_load():
    logger = logging.getLogger(name=__name__)
    logger.setLevel(logging.INFO)

    trained_model = os.path.expanduser('~/.EasyOCR/model/craft_mlt_25k.pth')
    net = easyocr.craft.CRAFT()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Pytorch Device was set {device}')

    loaded = torch.load(f=trained_model, map_location=device)
    # print(type(loaded)) -> <class 'collections.OrderedDict'>
    print(loaded)


if __name__ == '__main__':
    example_torch_load()