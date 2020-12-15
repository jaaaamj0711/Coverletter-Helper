from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
import torch

# Data set 함수 정의
def dataset (file_path):
  data = []
  tokenizer = SentencepieceTokenizer(get_tokenizer())
  f = open(file_path, 'r', encoding='utf-8')

  while True:
    file = f.readline()

    if not file:
      break
    line = tokenizer(file[:-1])
    indexing_word = [vocab[vocab.bos_token]]+ vocab[line] + [vocab[vocab.eos_token]]
    data.append(indexing_word)

  f.close()

  return data

model, vocab = get_pytorch_kogpt2_model()

# 경로 지정 및 불러오기
load_path = 'C:/Users/user/KoGPT2/KoGPT2_checkpoint.tar'
checkpoint = torch.load(load_path, map_location=torch.device(PU))

model.to(torch.device(PU)) 
torch.load(load_path, map_location=torch.device(PU))

model.eval()

# 저장 경로 지정
save_path= 'C:/Users/user/KoGPT2/'

# 파라미터 설정
kogpt2_config = {
      "initializer_range": 0.02,
      "layer_norm_epsilon": 0.000025,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12,
      "n_positions": 1024,
      "vocab_size": 50000
}

# check point
checkpoint = torch.load(save_path+'KoGPT2_checkpoint.tar', map_location=PU)

# Modeling
kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
kogpt2model.eval()
kogpt2model.to(torch.device(PU))

model = kogpt2model

# 토크나이저 정의
Tokenizer = SentencepieceTokenizer(get_tokenizer(), num_best=0, alpha=0)

# 생성할 문장의 키워드 설정 및 파라미터 조정
sentence = '빅데이터'
toked = Tokenizer(sentence)
temp = []
cnt = 0

# 결과 출력
while True:
  input_ids = torch.tensor([vocab[vocab.bos_token],] + vocab[toked]).unsqueeze(0)
  pred = model(input_ids)[0]

  gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())
  print(gen)
  print(gen[-1])
  gen = gen[-1]
  cnt += 1

  if cnt == 50:
    break

  if '</s>' == gen:
    break
  sentence += gen.replace('▁', ' ')
  toked = Tokenizer(sentence)