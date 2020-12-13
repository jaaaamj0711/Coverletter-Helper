def dataset (file_path):
  data = []
  tokenizer = SentencepieceTokenizer(get_tokenizer())
  f = open(file_path,'r',encoding='utf-8')

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

load_path = 'C:/Users/user/KoGPT2/KoGPT2_checkpoint.tar'
checkpoint = torch.load(load_path, map_location=torch.device(PU))


model.to(torch.device(PU)) #모델 연산 유닛 설정
torch.load(load_path, map_location=torch.device(PU))

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

