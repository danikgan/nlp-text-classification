# -*- coding: utf-8 -*-

# import from pytorch
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse

# import from folders
#from DPCNN.config import Config
#from DPCNN.model import DPCNN
#from DPCNN.data import TextDataset

from config import Config
from model import DPCNN
from data import TextDataset

# seed for random numbers
torch.manual_seed(1)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out_channel', type=int, default=2)
parser.add_argument('--label_num', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
# used for this:
# python main.py --lr=0.001 --epoch=20 --batch_size=64 --gpu=0 --seed=0 --label_num=2

# seed for random numbers
torch.manual_seed(args.seed)

# enable GPU
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

# Create the configuration
config = Config(sentence_max_size=50,
                batch_size=args.batch_size,
                word_num=11000,
                learning_rate=args.lr,
                epoch=args.epoch,
                cuda=args.gpu,
                label_num=args.label_num,
                out_channel=args.out_channel)

training_set = TextDataset(path='data/train')

training_iter = data.DataLoader(dataset=training_set,
                                batch_size=config.batch_size,
                                num_workers=2)


model = DPCNN(config)
embeds = nn.Embedding(config.word_num, config.word_embedding_dimension)

if torch.cuda.is_available():
    model.cuda()
    embeds = embeds.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr)

count = 0
loss_sum = 0

# Train the model
for epoch in range(config.epoch):
    for data, label in training_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            labels = label.cuda()

        input_data = embeds(autograd.Variable(data))
        out = model(data)
        loss = criterion(out, autograd.Variable(label.float()))

        loss_sum += loss.data[0]
        count += 1

        if count % 100 == 0:
            print("epoch", epoch, end='  ')
            print("The loss is: %.5f" % (loss_sum/100))

            loss_sum = 0
            count = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # save the model in every epoch
    model.save('checkpoints/epoch{}.ckpt'.format(epoch))

