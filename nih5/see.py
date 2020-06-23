import pandas as pd

#data = pd.read_csv('./worker0/train.csv')
#labels = list(data['label'].unique())
#labels.sort()

for i in range(3):
    print(f"Worker #{i} -----------")
    d = pd.read_csv(f"./worker{i}/train.csv")
    labels = d['label'].unique()
    for l in labels:
        print(l, d[d['label']==l]['label'].count())

