with open('data/atis.test.iob','r') as in_f:
    with open('data/test.seq.in', 'w') as out_f,open('data/test.label', 'w') as out_label_f,open('data/test.seq.out', 'w') as out_tag_f: 
        for line in in_f.readlines():
            out_label_f.write('test\n')
            for w in line.split()[0:-1]:
                out_f.write(w+' ')
                out_tag_f.write('BOS'+' ')
            out_f.write('\n')
            out_tag_f.write('\n')

with open('./data/atis.train.w-intent.iob') as in_f:
    with open('./data/train.label','w') as label_f, open('./data/train.seq.in','w') as seq_f, open('./data/train.seq.out','w') as seq_out_f:
        with open('./data/dev.label','w') as d_label_f, open('./data/dev.seq.in','w') as d_seq_f, open('./data/dev.seq.out','w') as d_seq_out_f:
            lines = in_f.readlines()
            for line in lines[:-1]:
                line = line.split('EOS')
                for w in line[0].split()[0:]:
                    seq_f.write(w+' ')
                seq_f.write('\n')
                for w in line[1].split()[0:-1]:
                    seq_out_f.write(w+' ')
                seq_out_f.write('\n')
                label_f.write(line[1].split()[-1])
                label_f.write('\n')
            for line in lines[-1:]:
                line = line.split('EOS')
                for w in line[0].split()[0:]:
                    d_seq_f.write(w+' ')
                d_seq_f.write('\n')
                for w in line[1].split()[0:-1]:
                    d_seq_out_f.write(w+' ')
                d_seq_out_f.write('\n')
                d_label_f.write(line[1].split()[-1])
                d_label_f.write('\n')
