import torch
import math


def main():
    x = torch.linspace(start=-math.pi, end=math.pi, steps=2000)
    y = torch.sin(x)

    p = torch.tensor(data=[1, 2, 3])
    xx: torch.tensor = x.unsqueeze(-1).pow(p)

    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=3, out_features=1),
        torch.nn.Flatten(start_dim=0, end_dim=1)
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')

    LEARNING_RATE = 1e-3
    # optim pakage를 사용해서 weights를 갱신해줄 Optimizer를 정의하자
    # tensorflow처럼 pytorch도 optim pakage에 여러 optimization algorithms을 포함하고 있는데
    # 지금은 RMSprop algorithm을 사용한다.
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=LEARNING_RATE)
    for t in range(2000):
        y_pred = model(xx)

        loss = loss_fn(input=y_pred, target=y)
        if t % 100 == 99:
            print(f'{t+1}번째 / Loss : {loss.item():.2f}')

        model.zero_grad()

        loss.backward()

        # step이라는 함수를 호출하면 params에 전달한 가중치가 갱신된다.
        optimizer.step()

    linear_layer = model[0]

    print(LEARNING_RATE)
    print(f'Result: y = {linear_layer.bias.item():.4f} + {linear_layer.weight[:, 0].item():.4f} x \
            + {linear_layer.weight[:, 1].item():.4f} x^2 + {linear_layer.weight[:, 2].item():.4f} x^3')


if __name__ == '__main__':
    main()