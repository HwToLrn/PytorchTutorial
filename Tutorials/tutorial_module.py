import torch
import math


# Module class를 상속받는다.
class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(size=()))
        self.b = torch.nn.Parameter(torch.randn(size=()))
        self.c = torch.nn.Parameter(torch.randn(size=()))
        self.d = torch.nn.Parameter(torch.randn(size=()))

    def forward(self, x):
        return self.a + self.b*x + self.c*(x**2) + self.d*(x**3)

    def string(self):
        return f'y = {self.a.item():.4f} + {self.b.item():.4f} x + {self.c.item():.4f} x^2 + {self.d.item():.4f} x^3'


def main():
    x = torch.linspace(start=-math.pi, end=math.pi, steps=2000)
    y = torch.sin(input=x)

    # 위에서 정의한 클래스로 모델을 생성합니다.
    model = Polynomial3()

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-6)
    for t in range(2000):
        # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
        y_pred = model(x)

        loss = loss_fn(input=y_pred, target=y)
        if t % 100 == 99:
            print(f'{t+1}번째\t/ Loss : {loss.item():.2f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Result: {model.string()}')


if __name__ == '__main__':
    main()
