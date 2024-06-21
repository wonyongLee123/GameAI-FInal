import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from joblib import dump

# 정제된 파일 읽기
file_path = 'data/cleaned_dataset.csv'
data_set = pd.read_csv(file_path, encoding='utf-8')

# sentences와 scores 분리
sentences = data_set['sentence'].tolist()
scores = data_set['score'].tolist()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences).toarray()
y = np.array(scores)

# Custom Dataset 정의
class PolitenessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = PolitenessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델 정의
class PolitenessModel(nn.Module):
    def __init__(self, input_dim):
        super(PolitenessModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

input_dim = X.shape[1]
model = PolitenessModel(input_dim)

# 학습 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 함수 정의
def train(model, dataloader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

train(model, dataloader, criterion, optimizer)

# 모델 결과 출력 함수
def predict(model, sentence, vectorizer):
    model.eval()
    with torch.no_grad():
        X = vectorizer.transform([sentence]).toarray()
        X = torch.tensor(X, dtype=torch.float32)
        output = model(X)
        return output.item()

# 테스트 문장 예측
test_sentences = [
    "Could you help me?",
    "Do this now!",
    "Please assist me with this.",
    "Complete this task immediately."
]

for sentence in test_sentences:
    score = predict(model, sentence, vectorizer)
    print(f"Sentence: {sentence} -> Politeness Score: {score:.4f}")

torch.save({
    'input_dim': input_dim,
    'model_state_dict': model.state_dict()
}, 'model.pth')

dump(vectorizer, 'count_vectorizer.joblib')