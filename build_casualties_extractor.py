from extractor import *
import stanfordnlp
import pandas as pd
stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline()

cas_df=pd.read_csv('sample_data\casualties.csv',sep='\t')
cas_df['title_p']=cas_df.apply(lambda x:x['title'].replace(x['death'],'Dperson') if not pd.isna(x),axis=1)
cas_df['title_p']=cas_df.apply(lambda x:x['title_p'].replace(x['wounded'],'Wperson') if not pd.isna(x),axis=1)
cas_df['title_p']=cas_df['title_p'].map(lambda x:gettext(x)[1])

count={}
death={}
wounded={}
for t in csa_df['title']:
    doc = nlp(text)
    for i in doc.sentences[0]:
        if i[2].text=='person' or i[2].text=='Dperson' or i[2].text=='Wperson':
            if i[0].text+'_'+i[1] in count.keys():
                count[i[0].text+'_'+i[1]]+=1
            else:
                count[i[0].text+'_'+i[1]]=1
            if i[2].text=='Dperson':
                if i[0].text+'_'+i[1] in count.keys():
                    death[i[0].text+'_'+i[1]]+=1
                else:
                    death[i[0].text+'_'+i[1]]=1
            if i[2].text=='Wperson':
                if i[0].text+'_'+i[1] in count.keys():
                    wounded[i[0].text+'_'+i[1]]+=1
                else:
                    wounded[i[0].text+'_'+i[1]]=1
        elif i[0].text=='person' or i[0].text=='Dperson' or i[0].text=='Wperson':
            if i[1]+'_'+i[2].text in count.keys():
                count[i[1]+'_'+i[2].text]+=1
            else:
                count[i[1]+'_'+i[2].text]=1
            if i[2].text=='Dperson':
                if i[1]+'_'+i[2].text in count.keys():
                    death[i[1]+'_'+i[2].text]+=1
                else:
                    death[i[1]+'_'+i[2].text]=1
            if i[2].text=='Wperson':
                if i[1]+'_'+i[2].text in count.keys():
                    wounded[i[1]+'_'+i[2].text]+=1
                else:
                    wounded[i[1]+'_'+i[2].text]=1

casualties={}
for k in death:
    if death[k]/count[k]>0.8:
        casualties[k]='d'
    elif wounded[k]/count[k]>0.8:
        wounded[k]='w'
out=open('probablity.json','w')
out.write(str(casualties))
out.close()