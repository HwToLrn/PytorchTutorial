import torch
import math


def main():
    x = torch.linspace(start=-math.pi, end=math.pi, steps=2000)
    y = torch.sin(x)

    # 예시로, y 값은 (x, x^2, x^3)와 같은 선형 함수라고 가정한다.
    # 이 y를 선형층 신경망(linear layer neural network)으로 생각할 수 있다.
    # tensor (x, x^2, x^3)을 만들어 보자.
    p = torch.tensor(data=[1, 2, 3])
    # Examples/explore_unsqueeze.py에서 예시를 보고 오자.
    xx: torch.tensor = x.unsqueeze(-1).pow(p)
    # x.shape : (2000, ) / x.unsqueeze(-1).shape : (2000, 1)
    # 아래 코드는 pow(p)에 의해 broadcasting이 되서 xx.shape은 (2000, 3)이 된다.
    # Ex) row1 : [5.x] -> [5.x, 25.x, 125.x] => shape : (1, 3)
    #     row2 : [10.x] -> [10.x, 100.x, 1000.x] => shape : (1, 3)
    # ... row2000까지 모두 xx에 들어가서 shape이 (2000, 3)이 만들어진다.
    # broadcasting 개념에 대해 모른다면 아래 링크를 보고 개념을 익히자
    # https://076923.github.io/posts/Python-numpy-12/
    # 참고로, numpy든 tensor든 자료형이 중요한 게 아니라 개념을 익히는 게 목적이다.

    # nn pakage를 사용해서 model을 정의한다.
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=3, out_features=1),
        torch.nn.Flatten(start_dim=0, end_dim=1)
    )

    # nn pakage는 흔히 알려진 손실 함수도 포함하고 있다.
    # 지금은 손실함수로 MSE를 사용한다.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    LEARNING_RATE = 1e-6
    for t in range(2000):
        # 순전파 모델에 xx값을 넣어서 예측값을 출력
        y_pred = model(xx)

        # Error 값 계산
        loss = loss_fn(input=y_pred, target=y)
        if t % 100 == 99:
            print(f'{t+1}번째 / Loss : {loss.item():.2f}')

        # 역전파를 하기 전에 기울기 값을 0으로 초기화한다.
        model.zero_grad()

        # 역전파 계산
        loss.backward()

        # weights 갱신
        with torch.no_grad():
            for param in model.parameters():
                param -= LEARNING_RATE * param.grad

    # 모델 layer에 접근하는 방법
    # list를 사용할 때처럼 [index]로 접근할 수 있다.
    # 모델의 구조를 보고 싶다면 print(model)을 치면
    # tensorflow에서 사용하는 명렁어 model.summary()와 비슷한 구조를 볼 수 있다
    linear_layer = model[0]

    # 첫 번째 layer인 linear layer의 weights와 biases를 출력한다.
    print(f'Result: y = {linear_layer.bias.item():.4f} + {linear_layer.weight[:, 0].item():.4f} x \
            + {linear_layer.weight[:, 1].item():.4f} x^2 + {linear_layer.weight[:, 2].item():.4f} x^3')


if __name__ == '__main__':
    main()