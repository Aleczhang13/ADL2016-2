import sys

raw_data = open(sys.argv[1],'r').readlines()
stopword = open('stopword','r').read().split()

    
f = open('pre_corpus','w')
i = 0
for line in raw_data:
    for w in line.split():
        if w in stopword:
            f.write(w+str(i)+' ')
            i+=1
        else:
            f.write(w+' ')
    f.write('\n')
f.close()
    
