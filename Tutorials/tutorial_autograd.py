import torch
import math


def main():
    dtype = torch.float
    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    # x의 범위 : -3.14 ~ 3.14, 2000개 생성
    x = torch.linspace(start=-math.pi, end=math.pi,
                       steps=2000, device=device, dtype=dtype)
    # y = sin(x) 수식
    y = torch.sin(x)

    # Randomly initialize weights
    # require_grad : 역전파를 수행하는 동안 자동으로 gradients를 계산하고 싶을 때 사용한다.
    a = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn(size=(), device=device, dtype=dtype, requires_grad=True)

    LEANRING_RATE = 1e-6
    for t in range(2000):
        # 순전파 : y값 예측
        y_pred = a + b*x + c*(x**2) + d*(x**3)

        # Compute and print Loss
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(f'{t+1}번째 / Loss : {loss.item():.2f}')

        # 역전파 : autograd
        # 손실함수 미분 및 갱신값을 위한 변수 계산
        # 손실함수가 (y_pred - y)**2 형태이므로 미분하면 (y_pred-y)에 2를 곱해준 형태가 된다.
        loss.backward()

        # 직접 기울기 감소량을 사용해서 weights 갱신한다.
        # torch.no_grad()로 감싸서 하는데 weights 값들이 requires_grad를 True로 해주었기에 가능한 방법이다.
        # 그러나 꼭 이런 방법으로 할 필요는 없다.
        # torch.no_grad() 설명
        #  - gradient 연산 옵션을 끌 때 사용한다.
        #  - 내부에서 새로 생성된 tensor들은 requires_grad=False 상태가 된다. -> 메모리 사용량을 save
        #  - 파이썬의 decorater로도 사용 가능
        with torch.no_grad():
            a -= LEANRING_RATE * a.grad
            b -= LEANRING_RATE * b.grad
            c -= LEANRING_RATE * c.grad
            d -= LEANRING_RATE * d.grad

            # 이후 weights 갱신을 위해 수동으로 기울기 값을 zero로 넣어준다.
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


if __name__ == '__main__':
    main()