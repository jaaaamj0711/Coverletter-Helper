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
