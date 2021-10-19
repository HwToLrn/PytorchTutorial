import torch
import math
from typing import Union


def example_shape_dim():
    x = torch.linspace(start=-math.pi, end=math.pi, steps=2000)
    print(x.shape, x.dim())  # -> torch.Size([2000]) 1
    p = torch.tensor(data=[1, 2, 3])
    print(p.shape, p.dim())  # -> torch.Size([3]) 1


# 여기서 Union이 의미하는 것은 return으로 int or float or tensor type이 가능함을 말한다.
def example_union() -> Union[int, float, torch.tensor]:
    int_data: int = 1
    float_data: float = 1.1
    tensor_data: torch.tensor = torch.tensor(data=[1., 2., 3.], dtype=torch.float32)
    # return float_data
    # return tensor_data
    return int_data


def main():
    x = torch.tensor(data=[[1, 2, 3, 4],
                           [5, 6, 7, 8]])
    print('Origin x :\n', x)
    print('Origin x shape : ', x.shape)  # x.size()
    print('\n')

    # torch.unsqueeze(dim: Any) - 1차원 => 2차원, 2차원 => 3차원
    #  - 현재 차원에서 +1을 해준 차원이 된다.

    # 직접 실행시키지 않을 거라면 '# shape:' 부분에서 '1'의 값의 위치와 dim=Union[-1, 0, 1, 2] 값에 주목하자.
    # '1'의 값의 위치는 생성된 차원 부분을 의미함을 알 수 있다.
    print('x.unsqueeze(0) :\n', x.unsqueeze(dim=0), x.unsqueeze(dim=0).shape, x.unsqueeze(dim=0).dim())  # shape:[1,2,4]
    print('x.unsqueeze(1) :\n', x.unsqueeze(dim=1), x.unsqueeze(dim=1).shape, x.unsqueeze(dim=1).dim())  # shape:[2,1,4]
    print('x.unsqueeze(2) :\n', x.unsqueeze(dim=2), x.unsqueeze(dim=2).shape, x.unsqueeze(dim=2).dim())  # shape:[2,4,1]
    # torch.unsqueeze(dim=-1)에서 -1은 마지막 차원에 추가하겠다는 의미
    print('x.unsqueeze(-1) :\n', x.unsqueeze(dim=-1), x.unsqueeze(dim=-1).shape, x.unsqueeze(dim=-1).dim())  # shape:[2,4,1]


if __name__ == '__main__':
    main()
