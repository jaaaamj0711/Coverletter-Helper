from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
import torch

# 연산자 설정
PU = 'cpu' 

# Data set class 정의
class Data_Set(Dataset):

  def __init__ (self, file_path, vocab, tokenizer):
    self.data = []
    self.vocab = vocab
    self.tokenizer = tokenizer

    f = open(file_path, 'r', encoding='utf-8')

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

#모델 연산 유닛 설정 및 모델 학습모드로 변경
model.to(torch.device(PU)) 
model.train() 

# 저장 경로 설정
save_path = 'C:/Users/user/KoGPT2/'

# Parameter 설정
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


# 체크 포인트 생성 및 모델 학습 준비
checkpoint = torch.load(save_path+'KoGPT2_checkpoint.tar', map_location=PU)
kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
kogpt2model.load_state_dict(checkpoint['model_state_dict'])
kogpt2model.train()
kogpt2model.to(torch.device(PU))

model = kogpt2model

#모델의 가중치 값을 저장하는 코드입니다.
torch.save(model.state_dict,save_path+'KoGPT2_checkpoint.tar')
model.load_state_dict(torch.load(save_path+'KoGPT2_checkpoint.tar')) 

# 모델 전체 저장
torch.save(model, 'C:/Users/user/KoGPT2/KoGPT2_checkpoint.tar') 

# 모델 불러오기
model = torch.load('C:/Users/user/KoGPT2/KoGPT2_checkpoint.tar') 

# file path 설정
file_path = 'C:/Users/user/KoGPT2/dataset.txt'
tokenizer = SentencepieceTokenizer(get_tokenizer(), num_best=0, alpha=0)

# 학습 데이터 set 불러오기
data = Data_Set(file_path, vocab, tokenizer)
dataset = DataLoader(data, batch_size=2, shuffle=True, pin_memory=True)

# 파라미터 설정
learning_rate = 0.00005
epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습
for epoch in range(checkpoint['epoch'], epochs+1):
  cnt = 0

  for data in dataset:
    optimizer.zero_grad()

    data = torch.stack(data)
    data = data.transpose(1, 0)
    data = data.to(PU)

    output = model(data,labels=data)
    loss, logits = output[:2]
    loss.backward()
    optimizer.step()

    if cnt % 20 == 0:
      print("[+] epoch : {}, cnt : {}, loss : {} [+]".format(epoch, cnt+1, str(loss)[7:12]))

    if epoch % 20 == 0 and cnt == 1:
      torch.save({
          'epoch': epoch,
          'cnt': cnt,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss,
          }, save_path+'KoGPT2_checkpoint.tar')
      
    cnt += 1

