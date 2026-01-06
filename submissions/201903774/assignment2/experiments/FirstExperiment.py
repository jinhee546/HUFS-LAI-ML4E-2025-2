from torch import nn

print("1st Experiment test")

#실험 1: 하이퍼파라미터 튜닝 - 은닉층 크기 증가(hidden_size1: 100 -> 550), 배치 크기 증가(batch_size: 128 -> 148), 에포크 수 증가(nb_epochs: 3 -> 4)


class firstMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=550, num_classes=10): # 은닉층 크기 증가(hidden_size1: 100 -> 550)
        super(firstMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        순전파 함수 // forward propagation
        x: 입력 텐서 (batch_size, 784)
        return: 출력 텐서 (batch_size, 10)
        """
        return self.layers(x)

# 하이퍼파라미터 설정
batch_size_1st = 148        # 배치 크기 증가(batch_size: 128 -> 148)
test_batch_size_1st = 1000  # 테스트 배치 크기 (메모리 효율을 위해 크게 설정)
learning_rate_1st = 1e-3    # 학습률 (0.001)
nb_epochs_1st = 4           # 에포크 수 증가(nb_epochs: 3 -> 4)