import torch


def pow():
    # 직관적으로 이해할 수 있도록 random 값보다는 일정한 정수값을 사용해서 확인하자.
    # a: torch.tensor = torch.randn(4)
    # b: torch.tensor = torch.randn(4)
    a: torch.tensor = torch.arange(start=2, end=6, dtype=torch.int32)
    b: torch.tensor = torch.arange(start=0, end=4, dtype=torch.int32)
    print('A : ', a)
    print('A.pow(2) : ', torch.pow(input=a, exponent=2))
    print('B : ', b)
    print('B.pow(2) : ', torch.pow(input=b, exponent=2))

    value = (b - a).pow(2)
    print('(B - A).pow(2)', value)

    return value

def main():
    value: torch.tensor = pow()

    # item method는 값이 1개이여야 python scalar 값으로 변환해준다.
    # 그래서 return type은 torch.tensor가 아니라 int인 것을 확인할 수 있다.
    # value.item()을 입력하고 ValueError message를 확인해보자.
    value_item: int = value.sum().item()
    print('\nvalue_item tpye : ', type(value_item))
    print('value_item value : ', value_item)


if __name__ == '__main__':
    main()