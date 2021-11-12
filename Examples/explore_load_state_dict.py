import collections
import torch
import os
import easyocr
import logging


def example_copyStateDict(state_dict):
    # print(list(state_dict.keys()))  # -> ['module.basenet.slice1.0.weight', 'module.basenet.slice1.0.bias', ...]
    # startswith('시작하는 문자', '시작 지점') : 특정 문자열 찾기
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        # key name : 'module.basenet.slice1.0.weight' -> 'basenet.slice1.0.weight'
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def example_torch_load():
    # logger = logging.getLogger(name=__name__)
    # logger.setLevel(logging.INFO)

    trained_model = os.path.expanduser('~/.EasyOCR/model/craft_mlt_25k.pth')
    net = easyocr.craft.CRAFT()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Pytorch Device was set : {device.upper()}')

    loaded: collections.OrderedDict = torch.load(f=trained_model, map_location=device)
    # print(type(loaded))  # -> <class 'collections.OrderedDict'>
    # print(loaded)  # -> 사전 학습된 각 layer의 w, b값이 출력
    new_loaded = example_copyStateDict(state_dict=loaded)
    net.load_state_dict(state_dict=new_loaded)
    print(net)  #  신경망 구조 확인


if __name__ == '__main__':
    example_torch_load()