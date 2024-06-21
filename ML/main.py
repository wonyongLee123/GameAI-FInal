import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import re
import pickle

data_set = pd.read_csv('data/cleaned_dataset.csv', encoding='utf-8')
print(data_set.head())
data_set.dropna()

sentences = data_set['sentence'].tolist()
scores = data_set['score'].tolist()

def remove_special_characters(text):
    # 정규 표현식을 사용하여 특수문자 제거 (알파벳, 숫자, 공백 제외 모든 문자 제거)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 각 문장에 특수문자 제거 적용
sentences_cleaned = [remove_special_characters(sentence) for sentence in sentences]

# CountVectorizer를 사용하여 문장을 벡터화
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences_cleaned).toarray()
y = np.array(scores)

with open('countvectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
# Custom Dataset 정의
class PolitenessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


train_dataset = PolitenessDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = PolitenessDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

test_dataset = PolitenessDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class PolitenessModel(nn.Module):
    def __init__(self, input_dim):
        super(PolitenessModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

input_dim = X.shape[1] #27579
model = PolitenessModel(input_dim)


criterion = nn.MSELoss()
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)


def train(model, dataloader, criterion, optimizer, epochs, val_dataloader=None):
    for epoch in range(epochs):

        model.train()

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


        # Validation
        if val_dataloader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_dataloader:
                    val_outputs = model(X_val_batch)
                    val_loss = criterion(val_outputs, y_val_batch)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}')


# 학습
train(model, train_dataloader, criterion, optimizer, epochs=500, val_dataloader=val_dataloader)

# 테스트 함수 정의
def test(model, dataloader):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for X_test_batch, y_test_batch in dataloader:
            test_outputs = model(X_test_batch)
            test_loss = criterion(test_outputs, y_test_batch)
            test_losses.append(test_loss.item())
    avg_test_loss = np.mean(test_losses)
    print(f'Test Loss: {avg_test_loss:.4f}')

# 테스트
test(model, test_dataloader)

torch.save(model.state_dict(), 'politeness_model.pth')