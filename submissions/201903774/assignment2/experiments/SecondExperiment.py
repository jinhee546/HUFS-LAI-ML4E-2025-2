from torch import nn

print("2nd Experiment test")

#실험 2: 모델 구조 개선 - 실험 1에서 튜닝한 하이퍼파라미터 유지,
#은닉층 2개 추가로 3층 신경망 구성(hidden_size1=589, hidden_size2=256, hidden_size3=128),
#LeakyReLU 및 Tanh 적용(전자는 첫 번째와 세 번째, 후자는 두 번째에 적용),Dropout 모든 층에 적용(p=0.3),
#BatchNormalization 1d 모든 층에 적용
#해당 실험의 secondMLP 코드 작성에는 ChatGPT의 도움을 받아 작성함

class secondMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size1=589, hidden_size2=256, hidden_size3=128, num_classes=10):
        super(secondMLP, self).__init__()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),   # normalize layer outputs
            nn.LeakyReLU(),                 # activation
            #nn.ReLU(),
            nn.Dropout(p=0.3),         # 30% dropout

            # Second hidden layer
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.Tanh(),                 # try tanh here
            #nn.ReLU(),
            nn.Dropout(p=0.3),

            # Third hidden layer
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.LeakyReLU(),                 # back to ReLU
            #nn.ReLU(),
            nn.Dropout(p=0.3),

            # Output layer
            nn.Linear(hidden_size3, num_classes)
        )

    def forward(self, x):
        """
        순전파 함수 // forward propagation
        x: 입력 텐서 (batch_size, 784)
        return: 출력 텐서 (batch_size, 10)
        """
        return self.layers(x)
