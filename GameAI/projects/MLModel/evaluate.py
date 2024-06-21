import torch
import torch.nn as nn
import re
import torch
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
import os

class PolitenessModel(nn.Module):
    def __init__(self, input_dim):
        super(PolitenessModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# load deep learning path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pth_path = os.path.join(base_dir, 'MLModel', 'model.pth')
vecotor_dict_path = os.path.join(base_dir, 'MLModel','count_vectorizer.joblib')

# apply
checkpoint = torch.load(pth_path)
dim = checkpoint['input_dim']

model = PolitenessModel(input_dim=dim)
model.load_state_dict(checkpoint['model_state_dict'])

vectorizer = CountVectorizer()

vectorizer = load(vecotor_dict_path)




def useModel(input):
    model.eval()
    vectorized_input = vectorizer.transform([input])
    tensored_input = torch.tensor(vectorized_input.toarray(), dtype=torch.float32)
    output = model(tensored_input)
    return round(output.item(), 4)



def main():
    text = str(input('text: '))
    out = useModel(text)

    print(f"point: {out}")
    
if __name__ == '__main__':
    main()
   