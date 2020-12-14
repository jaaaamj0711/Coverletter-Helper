from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch





  
PU='cpu' 


class Data_Set(Dataset):

  def __init__ (self, file_path,vocab,tokenizer):
    self.data = []
    self.vocab = vocab
    self.tokenizer = tokenizer

    f = open(file_path,'r',encoding='utf-8')

    file = f.read()
    file = file.split('\n')

    dataset = []
    now = ''
    for i, line in enumerate(file):
      if i % 30 == 0 and i != 0:
        dataset.append(now)
        now = ''

      now = now + '\n' + line

    for line in dataset:
      tokenized_line = tokenizer(line[:-1])

      indexing_word = [vocab[vocab.bos_token], ]+ vocab[tokenized_line] + [vocab[vocab.eos_token]]
      self.data.append(indexing_word)

    f.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]


model, vocab = get_pytorch_kogpt2_model()

model.to(torch.device(PU)) #모델 연산 유닛 설정
model.train() #모델 학습모드로 변경

