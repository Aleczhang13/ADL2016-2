import json
import sys
import numpy as np
import Levenshtein
method='levenshtein'
# read data start
# sys.argv[1] => the output of NN (the file boboRay send to Chi)
# sys.argv[2] => the test file TA give us, named "testing_data.json" 
# sys.argv[3] => the output answer name
with open(sys.argv[1],'r') as fin:
	answer_data=json.load(fin) 
with open(sys.argv[2],'r') as fin:
	temp=json.load(fin)
choice_list=list()
for x in temp:
	choice_list.append(x['answer_list'])
# read data end

# print (choice_list)
def choose_one(answer_data, choice_list):
	result=''
	for i in range(len(answer_data)-1):
		answer=answer_data[str(i)]
		choice=choice_list[i]
		result+=str(find_most_number_included_word(answer,choice))+'\n'
	with open(sys.argv[3],'wt') as fout:
		fout.write(result[:-1])


def find_most_number_included_word(answer,choice):
	# ARGUMENT || TYPE || DESCRIPTION
	# answer || string || the question's answer
	# choice || list || the choice of the question
	predict=list()
	if method == 'naive_comparison':
		for x in choice:
			predict.append(naive_comparison(answer,x))
	elif method == 'levenshtein':
		for x in choice:
			predict.append( Levenshtein.ratio(answer,x))
	# print(predict)
	return predict.index(max(predict))
	


def naive_comparison(answer, choice):
	# ARGUMENT || TYPE || DESCRIPTION
	# answer || string || the question's answer
	# choice || string || one choice of the question
	count_include=0
	choice_list = choice.split()
	answer_list = answer.split()
	for word in choice_list:
		if word in answer_list:
			count_include+=1
	# print(answer, choice, count_include)
	return count_include

if __name__ == "__main__":
	choose_one(answer_data, choice_list)
	pass