import json

train_f = open('./data/NLG_data/train.json','r')
valid_f = open('./data/NLG_data/valid.json','r')

train = json.load(train_f)
valid = json.load(valid_f)

with open('./data/NLG_data/train.in','w') as in_f, open('./data/NLG_data/train.out','w') as out_f:
    for line in train:
        in_f.write(line[0]+'\n')
        in_f.write(line[0]+'\n')
        out_f.write(line[1]+'\n')
        out_f.write(line[2]+'\n')

with open('./data/NLG_data/valid.in','w') as in_f, open('./data/NLG_data/valid.out','w') as out_f:
    for line in valid:
        in_f.write(line[0]+'\n')
        in_f.write(line[0]+'\n')
        out_f.write(line[1]+'\n')
        out_f.write(line[2]+'\n')


