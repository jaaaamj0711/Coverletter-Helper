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


save_path = 'C:/Users/user/KoGPT2/KoGPT2_checkpoint.tar'

from transformers import GPT2Config, GPT2LMHeadModel


save_path = 'C:/Users/user/KoGPT2/'

kogpt2_config = {
      "initializer_range": 0.02,
      "layer_norm_epsilon": 0.000001,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12,
      "n_positions": 1024,
      "vocab_size": 50000,
      "activation_function": "gelu"
}


checkpoint = torch.load(save_path+'KoGPT2_checkpoint.tar', map_location=PU)

kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

kogpt2model.load_state_dict(checkpoint['model_state_dict'])

kogpt2model.train()

kogpt2model.to(torch.device(PU))

model = kogpt2model


torch.save(model.state_dict,save_path+'KoGPT2_checkpoint.tar') #모델의 가중치 값을 저장하는 코드입니다.

model.load_state_dict(torch.load(save_path+'KoGPT2_checkpoint.tar')) #모델의 가중치 값을 불러오는 코드입니다.

torch.save(model, 'C:/Users/user/KoGPT2/KoGPT2_checkpoint.tar') #모델 전체를 저장하는 코드입니다.


model = torch.load('C:/Users/user/KoGPT2/KoGPT2_checkpoint.tar') #모델 전체를 불러오는 코드입니다.


file_path = 'C:/Users/user/KoGPT2/dataset.txt'

tokenizer = SentencepieceTokenizer(get_tokenizer(), num_best=0, alpha=0)

data = Data_Set(file_path, vocab, tokenizer)

dataset = DataLoader(data, batch_size=2, shuffle=True, pin_memory=True)

learning_rate = 0.00005
epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

