import json
import random
import sys

input_f = sys.argv[1]
output_f = sys.argv[2]
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


def get_para(article):
    answer_text = article['answer_list'][article['answer']]
    question = article['question']
    context = article['context']
    title = 'squad'
    version = '1.1'
    answers = get_answers(answer_text, context)
    idx = random.randint(1,1000000)
    while idx in idx_set:
        idx = random.randint(1,1000000)
    idx_set.add(idx)
    par_dict = {'title': 'title' + context[:12],
                'paragraphs':[{'context':context,
                              'qas':[{'id': idx,
                                      'question':question,
                                      'answers':answers}]}]}
    return par_dict


data = []
for art in inp[:]:
    para= get_para(art)
    data.append(para)
data_dict = {'version':'1.1',
            'data':data}

with open(output_f,w') as o_f:
    json.dump(data_dict,o_f)
