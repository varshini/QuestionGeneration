import random

random.seed(42)

f = open('../../data/sentlevel_squad_train.hashed')
outfile = open('../../data/sentlevel_squad_train.hashed.shuffle','w')
training_data = f.read().strip().split("\n\n")
index = []
for i in range(len(training_data)):
    index.append(i)

random.shuffle(index)
for i in index:
    outfile.write(training_data[i] + "\n\n")
