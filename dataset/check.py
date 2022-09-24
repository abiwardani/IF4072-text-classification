import pandas as pd

ps = pd.read_csv('dataset/dev.csv')
dtext_a = list(ps['text_a'])
ps = pd.read_csv('dataset/train.csv')
ttext_a = list(ps['text_a'])
tlabel = list(ps['label'])

text_a = []
label = []
count = 0
for t,l in zip(ttext_a, tlabel):
    if t in dtext_a:
        count += 1
    else:
        text_a.append(t)
        label.append(l)
if count:
    print(len(ttext_a), count, len(text_a))
    dic = {'text_a':text_a, 'label':label}
    ps = pd.DataFrame(dic)
    ps.to_csv('dataset/train_filtered.csv')

print(label[3])