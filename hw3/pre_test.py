import sys
test_path = sys.argv[1]

with open(test_path,'r') as in_f:
    with open('data/test.seq.in', 'w') as out_f,open('data/test.label', 'w') as out_label_f,open('data/test.seq.out', 'w') as out_tag_f: 
        for line in in_f.readlines():
            out_label_f.write('test\n')
            for w in line.split()[0:-1]:
                out_f.write(w+' ')
                out_tag_f.write('BOS'+' ')
            out_f.write('\n')
            out_tag_f.write('\n')
