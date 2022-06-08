import numpy as np
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from bert_embedder import BertEmbedder
import json
import torch

#load model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
config = AutoConfig.from_pretrained("bert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedder = BertEmbedder(config,maxlen=512).to(device)

#load reviews
#"./gorecdial_dict.json" is a dict using global as key
movie_review_path = "./gorecdial_dict.json"
with open(movie_review_path, 'r') as f:
    movie_reviews = json.load(f)

look_up_table = [] #store emebeddings
# for each of the 3782 movies sorted by global
for key in range(3782):
    accumulative = torch.zeros((1,768))
    movie_review = movie_reviews[str(key)]
    # if this movie has no review
    if movie_review["titles"] == None or len(movie_review["titles"]) == 0:
        look_up_table.append(accumulative)
        assert torch.all(torch.isfinite(accumulative))
        continue
    #amke sure the number of title is equal to the number of content
    assert len(movie_review["titles"]) == len(movie_review["content"])
    #generate embed for each review and add them up
    for i in range (len(movie_review["titles"])):
        batch = tokenizer.encode_plus(movie_review["titles"][i],movie_review["content"][i],return_tensors="pt",padding=True,truncation='only_second')
        embedding = embedder(batch.input_ids,batch.token_type_ids,batch.attention_mask).cpu()
        embedding = embedding.to(device=torch.device('cpu')) 
        accumulative = torch.add(accumulative, embedding)
    #take average
    item_embeddings = torch.div(accumulative, len(movie_review["titles"]))
    #append to embeding look up table 
    look_up_table.append(item_embeddings)
    #make sure all the numbers are finite
    assert torch.all(torch.isfinite(item_embeddings))
#reshape to (3782 *768)
look_up_table = torch.cat(look_up_table, dim=0)
#save the embdding tensor
torch.save(look_up_table, 'tensor_embed.pt')
