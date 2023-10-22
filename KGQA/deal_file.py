import os

outfile = open(os.path.join('preprocess/', 'formatted_question.txt'), 'w', encoding='utf-8')
for line in open(os.path.join('preprocess/', 'original_question.txt'), 'r', encoding='utf-8'):
    items = line.strip().split("|")
    if len(items) != 5:
        print(items)
        continue
    outfile.write("{}\t{}\t{}\t{}\t{}\n".format(items[0], items[1], items[2], items[3], items[4]))
outfile.close()