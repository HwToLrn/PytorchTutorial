import torch
import easyocr
import numpy as np
import cv2
import torch.nn.functional as F
import os
import collections
import matplotlib.pyplot as plt


class double_conv(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch + mid_ch, out_channels=mid_ch, kernel_size=(1, 1)),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv.to(self.device)

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_class(torch.nn.Module):
    def __init__(self):
        super(conv_class, self).__init__()
        num_class = 2
        self.conv_cls = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1), torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1), torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1), torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(16, 16, kernel_size=(1, 1)), torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(16, num_class, kernel_size=(1, 1)),
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv_cls.to(self.device)

    def forward(self, x):
        x = self.conv_cls(x)
        return x


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


# easyocr.craft.CRAFT() code comprehension
def main():
    # data
    image = '../ImageDatas/image1.jpg'  # shape : (243, 208, 3)
    image, _ = easyocr.utils.reformat_input(image=image)
    print('image.shape : ', image.shape)  # -> height, width, channel
    print('type(image) : ', type(image))

    # hyper parameters
    min_size = 20
    text_threshold = 0.7
    low_text = 0.4
    link_threshold = 0.4
    canvas_size = 2560
    mag_ratio = 1.
    slope_ths = 0.1
    ycenter_ths = 0.5
    height_ths = 0.5
    width_ths = 0.5
    add_margin = 0.1
    reformat = True
    optimal_num_chars = None
    estimate_num_chars = optimal_num_chars is not None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # network
    base_net = easyocr.model.modules.vgg16_bn(pretrained=False, freeze=False)
    base_net.to(device)
    # print(base_net)

    # make image's dimension same
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    # resize
    img_resized_list = []
    target_ratio = 0.
    for img in image_arrs:
        img_resized, target_ratio, _ = easyocr.imgproc.resize_aspect_ratio(img, canvas_size,
                                                                           interpolation=cv2.INTER_LINEAR,
                                                                           mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = [np.transpose(easyocr.imgproc.normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    source = base_net(x)
    # for idx in range(5):
    #     print(source[idx])
    # print()

    # U network
    upconv1 = double_conv(1024, 512, 256)
    upconv2 = double_conv(512, 256, 128)
    upconv3 = double_conv(256, 128, 64)
    upconv4 = double_conv(128, 64, 32)

    # torch.cat() : https://sanghyu.tistory.com/85
    print(source[0].shape, '\tsource[0].shape')  # torch.Size([1, 1024, 14, 16])
    print(source[1].shape, '\tsource[1].shape')  # torch.Size([1, 512, 14, 16])
    fst_unet = torch.cat(tensors=[source[0], source[1]], dim=1)  # dim=1: 2번째 차원
    fst_unet_out = upconv1(fst_unet)
    print(fst_unet.shape, '\tfst_unet shape')  # torch.Size([1, 1536, 14, 16])
    print(fst_unet_out.size(), '\tfst_unet_out.size()')
    print('='*50)

    # fst_unet과 source[2]의 shape을 맞추기 위해 F.interpolate()를 사용한다.
    print(source[2].size(), '\tsource[2].size()')  # torch.Size([1, 512, 28, 32])
    y = F.interpolate(input=fst_unet_out, size=source[2].size()[2:], mode='bilinear', align_corners=False)
    scd_unet = torch.cat(tensors=[y, source[2]], dim=1)
    scd_unet_out = upconv2(scd_unet)
    print(y.size(), '\ty.size()')
    print(scd_unet.size(), '\tscd_unet.size()')
    print(scd_unet_out.size(), '\tscd_unet_out.size()')
    print('='*50)

    print(source[3].size(), '\tsource[3].size()')  # torch.Size([1, 512, 28, 32])
    y = F.interpolate(input=scd_unet_out, size=source[3].size()[2:], mode='bilinear', align_corners=False)
    trd_unet = torch.cat(tensors=[y, source[3]], dim=1)
    trd_unet_out = upconv3(trd_unet)
    print(y.size(), '\ty.size()')
    print(trd_unet.size(), '\ttrd_unet.size()')
    print(trd_unet_out.size(), '\ttrd_unet_out.size()')
    print('='*50)

    print(source[4].size(), '\tsource[4].size()')  # torch.Size([1, 512, 28, 32])
    y = F.interpolate(input=trd_unet_out, size=source[4].size()[2:], mode='bilinear', align_corners=False)
    last_unet = torch.cat(tensors=[y, source[4]], dim=1)
    feature = upconv4(last_unet)
    print(y.size(), '\ty.size()')
    print(last_unet.size(), '\tlast_unet.size()')
    print(feature.size(), '\tfeature.size()')
    print('='*50)

    conv_cls = conv_class()
    y_out = conv_cls(feature)
    print(y_out.size(), '\ty_out.size()')
    y_out = y_out.permute(0, 2, 3, 1)  # shape : data num, height, width, channel
    print(y_out.size(), '\ty_out.size() permuted')
    print('='*50)

    trained_model = os.path.expanduser('~/.EasyOCR/model/craft_mlt_25k.pth')
    net = easyocr.craft.CRAFT()
    loaded: collections.OrderedDict = torch.load(f=trained_model, map_location=device)
    new_loaded = example_copyStateDict(state_dict=loaded)
    net.load_state_dict(state_dict=new_loaded)
    net.to(device)

    print("Origin network's results -------------------------")
    with torch.no_grad():
        y_out, feature = net(x)
        print(feature.size(), '\tfeature.size()')
        print(y_out.size(), '\ty_out.size() permuted')

    boxes_list, polys_list = [], []
    for out in y_out:
        # make score and link map
        # 각 tensor값들을 cpu 메모리에 data를 올려서 cpu에서 연산이 가능하도록 만들고 type은 numpy다
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # print(score_text.shape)  # shape : 112, 128
        # print(score_link.shape)  # shape : 112, 128

        # 이미지 임계처리 설명 참고 링크
        # https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html
        ret, text_score = cv2.threshold(src=score_text, thresh=low_text, maxval=1, type=0)
        ret, link_score = cv2.threshold(src=score_link, thresh=link_threshold, maxval=1, type=0)

        # 결과물 참고 이미지 : ImageDatas/image1_model_output.jpg
        plt.subplot(1, 2, 1)
        plt.title('Text score')
        plt.imshow(text_score)
        plt.subplot(1, 2, 2)
        plt.title('Link score')
        plt.imshow(link_score)
        plt.show()


        # Post-processing
        # boxes, polys, mapper = easyocr.craft_utils.getDetBoxes(
        #     textmap=score_text, linkmap=score_link,
        #     text_threshold=text_threshold, link_threshold=link_threshold,
        #     low_text=low_text, poly=False, estimate_num_chars=estimate_num_chars)


if __name__ == '__main__':
    main()