import torch
import easyocr


def main():
    base_net = easyocr.model.modules.vgg16_bn(pretrained=False, freeze=False)
    print(base_net)


if __name__ == '__main__':
    main()