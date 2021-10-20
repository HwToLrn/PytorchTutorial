import torch
import math

# 샌드리아스폐셜(best) 파마산 콜라 1개
# 휠렛핫칠리치킨 파마산 에그마요 1개
# 불갈비치즈 파마산 콜라 2개

def main():
    dtype = torch.float
    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    # x의 범위 : -3.14 ~ 3.14, 2000개 등분하여 2000개의 값 생성
    x = torch.linspace(start=-math.pi, end=math.pi,
                       steps=2000, device=device, dtype=dtype)
    # y = sin(x) 수식
    y = torch.sin(x)

    # Randomly initialize weights
    a = torch.randn(size=(), device=device, dtype=dtype)
    b = torch.randn(size=(), device=device, dtype=dtype)
    c = torch.randn(size=(), device=device, dtype=dtype)
    d = torch.randn(size=(), device=device, dtype=dtype)

    LEANRING_RATE = 1e-6

    for t in range(2000):
        # 순전파 : y값 예측
        y_pred = a + b*x + c*(x**2) + d*(x**3)

        # Compute and print Loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t+1, loss)

        # 역전파
        # 손실함수 미분 및 갱신값을 위한 변수 계산
        # 손실함수가 (y_pred - y)**2 형태이므로 미분하면 (y_pred-y)에 2를 곱해준 형태가 된다.
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * (x ** 2)).sum()
        grad_d = (grad_y_pred * (x ** 3)).sum()

        # weights 갱신
        a -= LEANRING_RATE * grad_a
        b -= LEANRING_RATE * grad_b
        c -= LEANRING_RATE * grad_c
        d -= LEANRING_RATE * grad_d

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

if __name__ == '__main__':
    main()