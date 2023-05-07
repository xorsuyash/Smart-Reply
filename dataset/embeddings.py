import os
import pickle  
import torch 
from torch.utils.data.dataloader import default_collate
from sent2vec.vectorizer import Vectorizer


os.chdir('/home/suyash/Desktop/Smart-Reply/dataset')
print(os.getcwd())

with open('input_texts.pickle','rb') as f :
    input_texts = pickle.load(f)


with open('target_texts.pickle','rb') as f:
    target_texts = pickle.load(f)

print(type(target_texts))


#input texts 
#target texts 

#embeddingd 
#class model 
#train 
#embeddings and labels

#class for deviding data into batches to prevent bottle neck of gpu 
class NaiveDataLoader:
    def __init__(self, dataset, batch_size=64, collate_fn=default_collate):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.index = 0
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            # stop iteration once index is out of bounds
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        return self.collate_fn([self.get() for _ in range(batch_size)])

    def get(self):
        item = self.dataset[self.index]
        self.index += 1
        return item
    




input_text_loader= NaiveDataLoader(input_texts,batch_size=500)
embeddings=[]


for sentence in input_text_loader:


    vectorizer=Vectorizer()
    vectorizer.run(sentence)

    vectors=vectorizer.vectors
    embeddings.append(vectors) 
