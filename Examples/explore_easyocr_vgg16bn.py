import torch
import easyocr
import numpy as np
import cv2
import torch.nn.functional as F


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
    y_out = torch.cat(tensors=[y, source[4]], dim=1)
    feature = upconv4(y_out)
    print(y.size(), '\ty.size()')
    print(y_out.size(), '\ty_out.size()')
    print(feature.size(), '\tfeature.size()')
    print('='*50)

    y_out = y_out.permute(0, 2, 3, 1)  # shape : data num, height, width, channel
    print(y_out.size(), '\ty_out.size()')


if __name__ == '__main__':
    main()