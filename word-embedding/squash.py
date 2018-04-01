fr=open('abstracts.EP.ascii.lower.txt','r')
fw=open('abstracts.EP.ascii.lower.tcbb.txt','w')
sumed=21424032
index=0
for line in fr:
    index+=1
    if index==sumed:
        break
    fw.write(line)
fw.flush()
fw.close()
