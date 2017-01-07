import json
import random
import sys
import os

input_f = sys.argv[1]
inp = json.loads(open(input_f,'r').read())
idx_set = set()

title = 'squad'
version = '1.1'

def get_answers(string, sentence):
    o_string = string[:]
    string = ' '+ string+ ' '
    indexs = []
    pre_idx = 0
    sub = sentence[:]
    while True:
        try:
            idx = sub.index(string)
            idx = idx+1
        except:
            break
        indexs.append(pre_idx + idx)
        pre_idx = pre_idx + idx+1
        sub = sentence[pre_idx:]
    answers = []
    if len(indexs) == 0:
        indexs.append(sentence.index(o_string))
    for idx in indexs:
        answers.append({'answer_start': idx,
                       'text':o_string})
    return answers
def get_para(article,idx):
    question = article['question']
    context = article['context']
    answer_text = context[:10].strip()
    title = 'squad'
    version = '1.1'
    answers = get_answers(answer_text, context)
    par_dict = {'title': 'title' + context[:12],
                'paragraphs':[{'context':context,
                              'qas':[{'id': idx,
                                      'question':question,
                                      'answers':answers}]}]}
    return par_dict

data = []
for idx, art in enumerate(inp[:]):
    para= get_para(art,idx)
    data.append(para)
data_dict = {'version':'1.1',
            'data':data}

if not os.path.exists('./data'):
    os.makedirs('./data')
with open('./data/test-v1.1.json','w') as o_f:
    json.dump(data_dict,o_f)
