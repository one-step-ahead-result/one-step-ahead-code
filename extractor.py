import re
import json
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import stanfordnlp
from collections import Counter
import pandas as pd
import difflib

stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline()

def isname(text):
    '''
    Find all words in the text where the first letter is capitalized
    '''
    word=text.split(' ')

    namelist=[]
    last=-2
    for i in range(len(word)):
        if word[i][0].isupper():
            if i==last+1:
                namelist[-1]=namelist[-1]+' '+word[i]
            else:
                namelist.append([word[i],i])
            last=i
    for i in namelist:
        text=text.replace(i[0],'$person-name')
    return namelist,text

def isperson(text):
    '''
    Use the wordnet to find words that represent people in text
    '''
    word=text.split(' ')
    personlist=[]
    temp=0
    text=' %s '%(text)
    for i in range(len(word)):
        #print(word[i])
        #print(personlist)
        if word[i]=='more':
            continue
        b=False
        for sys in [w for w in wordnet.synsets(word[i])]:
            if b:
                break
            for temlist in sys.hypernym_paths():
                if b:
                    break
                if wordnet.synset('person.n.01') in temlist :#or wordnet.synset('social_group.n.01') in temlist
                    b=True
                    add=False
                    if i!=0:
                        wn=[syn.name() for syn in wordnet.synsets(word[i-1])]
                        l=len(personlist)
                        for w in wn:
                            if '.a.' in w:
                                if len(personlist)>0 and personlist[-1][1]==i-1-temp:
                                    #print(111)
                                    personlist[-1][0]+=' '+word[i]
                                    l=l-1
                                    temp+=1
                                    add=True
                                else:
                                    #print(222)
                                    personlist.append([word[i-1]+' '+word[i],i-1-temp])
                                    temp+=1
                                    add=True
                                break
                    if not add:
                        #print(333)
                        personlist.append([word[i],i-temp])
    for i in personlist:
        text=text.replace(' '+i[0]+' ',' $person-personword ')
    return personlist,text[1:][:-1]

def ispersonword(word):
    for sys in wordnet.synsets(word):
        for temlist in sys.hypernym_paths():
            if wordnet.synset('person.n.01') in temlist or wordnet.synset('social_group.n.01') in temlist:
                return True
    return False

def intnub(text):
    '''
    Find all number in text
    '''
    word=text.split(' ')
    nublist=[]
    i=0
    temnub=0
    while i <len(word):
        if re.match('^[0-9]*$',word[i]) is not None:
            if i+1<len(word) and ispersonword(word[i+1]):
                nublist.append([word[i]+' '+word[i+1],i-temnub])
                temnub+=1
                text=text.replace(nublist[-1][0],'$person-intnub',1)
                i+=1
            elif i+2<len(word) and ispersonword(word[i+2]):
                nublist.append([word[i]+' '+word[i+1]+' '+word[i+2],i-temnub])
                temnub+=2
                text=text.replace(nublist[-1][0],'$person-intnub',1)
                i+=2
            else:
                nublist.append([word[i],i])
                text=text.replace(nublist[-1][0],'$person-intnub',1)
                
        i+=1
    return nublist,text

def strnub(text):
    '''
    Find all word that represent a numbers.
    '''
    doc = nlp(text)
    word=text.split(' ')
    strnublist=[]
    personlist=[]
    temnub=0
    for ent in doc.ents:
        if ent.label_=='CARDINAL':
            strnublist.append(ent.text)
    i=0
    while i <len(word):
        if word[i] in strnublist:
            if i+1<len(word) and ispersonword(word[i+1]):
                personlist.append([word[i]+' '+word[i+1],i-temnub])
                temnub+=1
                text=text.replace(personlist[-1][0],'$person-strnub',1)
                i+=1
            elif i+2<len(word) and ispersonword(word[i+2]):
                personlist.append([word[i]+' '+word[i+1]+' '+word[i+2],i-temnub])
                temnub+=2
                text=text.replace(personlist[-1][0],'$person-strnub',1)
                i+=2
            else:
                personlist.append([word[i],i-temnub])
                text=text.replace(personlist[-1][0],'$person-strnub',1)
        i+=1
    return personlist,text

def getpersonword(text):
    '''
    Find words that may represent people.
    '''
    intlist,text=intnub(text)
    
    strlist,text=strnub(text)
    
    #namelist,text=isname(text)
    
    personlist,text=isperson(text)
    
    word=text.split(' ')
    returnlist=[]
    for i in range(len(word)-1,-1,-1):
        if word[i]=='$person-strnub':
            returnlist.append([strlist.pop()[0],i])
        elif word[i]=='$person-intnub':
            returnlist.append([intlist.pop()[0],i])
        elif word[i]=='$person-personword':
            returnlist.append([personlist.pop()[0],i])
        #elif word[i]=='$person-name':
            #returnlist.append([namelist.pop()[0],i])
    return returnlist,text

def getneighbor(personlist,text):
    '''
    Merge adjacent words that represent people.
    '''
    word=text.split(' ')
    re=[]
    for personitem in personlist:
        temptec={}
        if personitem[1]-3>0:
            temptec['-3']=word[personitem[1]-3]
        else:
            temptec['-3']=None
        if personitem[1]-2>0:
            temptec['-2']=word[personitem[1]-2]
        else:
            temptec['-2']=None
        if personitem[1]-1>0:
            temptec['-1']=word[personitem[1]-1]
        else:
            temptec['-1']=None
        temptec['0']=personitem[0]
        if personitem[1]+1<len(word):
            temptec['1']=word[personitem[1]+1]
        else:
            temptec['1']=None
        if personitem[1]+2<len(word):
            temptec['2']=word[personitem[1]+2]
        else:
            temptec['2']=None
        if personitem[1]+3<len(word): 
            temptec['3']=word[personitem[1]+3]
        else:
            temptec['3']=None
        re.append(temptec)
    return re

def mergeword(personlist,text,otext):
    '''
    Merge adjacent words that represent people.
    '''
    otext=' %s '%(otext)
    word=text.split(' ')
    wordlist=['0' for i in range(len(word))]
    for i in personlist:
        wordlist[i[1]]=i[0]
    temp=0
    re=[]
    i=0
    while(i<len(word)):
        if i<len(word)-2 and wordlist[i]!='0' and wordlist[i+2]!='0' and word[i+1].lower()=='and':
            re.append([wordlist[i]+' and '+wordlist[i+2],i-temp])
            temp+=2#len((wordlist[i]+' and '+wordlist[i+2]).split(' '))-1
            otext=otext.replace(wordlist[i]+' '+word[i+1]+' '+wordlist[i+2],'person')
            i+=2
        elif i<len(word)-2 and wordlist[i]!='0' and wordlist[i+2]!='0' and word[i+1].lower()==',':
            re.append([wordlist[i]+' , '+wordlist[i+2],i-temp])
            temp+=2#len((wordlist[i]+' and '+wordlist[i+2]).split(' '))-1
            otext=otext.replace(wordlist[i]+' '+word[i+1]+' '+wordlist[i+2],'person')
            i+=2
        elif i<len(word)-1 and wordlist[i]!='0' and wordlist[i+1]!='0':
            re.append([wordlist[i]+' '+wordlist[i+1],i-temp])
            temp+=1#len((wordlist[i]+' and '+wordlist[i+1]).split(' '))-1
            otext=otext.replace(wordlist[i]+' '+wordlist[i+1],'person')
            i+=1
        
        elif wordlist[i]!='0':
            re.append([wordlist[i],i-temp])
            otext=otext.replace(' '+wordlist[i]+' ',' person ')
        i+=1
    otext=otext[1:][:-1]
    return re,otext

def dropstopword(text):
    '''
    Drop stop words.
    '''
    text=text.replace(' a ','1').replace(' an ','1')
    allword=text.split(' ')
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\'','|','-']
    stops = set(stopwords.words("english"))
    allword = [word.lower() for word in allword if word.lower() not in stops and word not in english_punctuations]
    return ' '.join(allword)


def getpersonneighbor(otext):
    #otext=dropstopword(otext)
    wordlist,text=getpersonword(otext)
    temtext=text
    wordlist,text=mergeword(wordlist,text,otext)
    while(temtext!=text):
        temtext=text
        wordlist,text=mergeword(wordlist,text,otext)
    return getneighbor(wordlist,text)

def gettext(otext):
    '''
    Get the text which replace the people word to 'person'.
    '''
    otext=' '+otext+' '
    otext=otext.replace(' one ',' 1 ').replace(' two ',' 2 ').replace(' three ',' 3 ').replace(' four ',' 4 ').replace(' five ',' 5 ')
    otext=otext.replace(' six ',' 6 ').replace(' seven ',' 7 ').replace(' eight ',' 8 ').replace(' nine ',' 9 ')
    otext=otext.replace(' One ',' 1 ').replace(' Two ',' 2 ').replace(' Three ',' 3 ').replace(' Four ',' 4 ').replace(' Five ',' 5 ')
    otext=otext.replace(' ten ',' 10 ').replace(' eleven ',' 11 ').replace(' twilve ',' 12 ').replace(' thriteen ',' 13 ').replace(' forteen ',' 14 ').replace(' fifteen ',' 15 ').replace(' sixteen ',' 16 ').replace(' seventeen ',' 17 ').replace(' eighteen ',' 18 ').replace(' nineteen ',' 19 ')
    otext=otext.replace(' Ten ',' 10 ').replace(' Eleven ',' 11 ').replace(' Twilve ',' 12 ').replace(' Thriteen ',' 13 ').replace(' Forteen ',' 14 ').replace(' Fifteen ',' 15 ').replace(' Sixteen ',' 16 ').replace(' Seventeen ',' 17 ').replace(' Eighteen ',' 18 ').replace(' Nineteen ',' 19 ')
    otext=otext.replace(' Six ',' 6 ').replace(' Seven ',' 7 ').replace(' Eight ',' 8 ').replace(' Nine ',' 9 ')
    otext=otext.replace('A ','1 ').replace('An ','1 ').replace(' a ',' 1 ').replace(' an ',' 1 ').replace("Dozen ",'12 ').replace(" Dozen",' 12').replace("dozen ",'12 ').replace(" dozen",' 12')
    otext=otext.replace('at least ','').replace('At least ','').replace('At Least ','').replace(', ',' , ').replace('  ,',' ,')
    otext=otext.replace('More than ','').replace('more than ','').replace('More Than ','')
    otext=otext[1:-1]
    wordlist,text=getpersonword(otext)
    temtext=text
    wordlist,text=mergeword(wordlist,text,otext)
    while(temtext!=text):
        temtext=text
        wordlist,text=mergeword(wordlist,text,otext)
    return wordlist,text 

def extract_casualties(text):
    '''
    Extract casualities from the text.
    '''
    probablity=eval(open('probablity.json').read())
    wordlist,text=gettext(text)
    doc = nlp(text)
    list_index=0
    death=None
    wounded=None
    for i in doc.sentences[0]:
        if i[2].text=='person':
            if i[0].text+'_'+i[1] in probablity.keys():
                if probablity[i[0].text+'_'+i[1]]=='d':
                    death=wordlist[list_index]
                else:
                    wounded=wordlist[list_index]
            list_index+=1
        elif i[0].text=='person':
            if i[1]+'_'+i[2].text in probablity.keys():
                if probablity[i[1]+'_'+i[2].text]=='d':
                    death=wordlist[list_index]
                else:
                    wounded=wordlist[list_index]
            list_index+=1
    if death is not None and len(re.findall('\d+',death))>0:
        death=re.findall('\d+',death)[0]
    elif death is not None:
        death=1
    else:
        death=0
    #print(wound)
    if wounded is not None and len(re.findall('\d+',wounded))>0:
        wounded=re.findall('\d+',wounded)[0]
    elif wounded is not None:
        wounded=1
    else:
        wounded=0
    return death,wounded

class Loaction_extactor():
    '''
    Class used to extract the location from the news.

    use the function 'getlocation' to extract the location from the news text.
    '''
    def __init__(self):
        self.location_data=pd.read_csv('allCountries.txt',sep='\t',names=['id','name','asciiname','alternatenames','latitude','longitude','feature class','feature code','country code','cc2','admin1 code','admin2 code','admin3 code','admin4 code','population','elevation','dem','timezone','modification date'])
        self.location_data=self.location_data.loc[self.location_data['feature class']=='A']
        self.country2code=pd.read_csv('country_code.csv',sep='\t')
        self.location_data['admin1 code']=self.location_data['admin1 code'].map(lambda x:str(x))
        self.location_data['statuscode']=self.location_data['country code']+'.'+self.location_data['admin1 code']

    def getlocationword(self,text):
        '''
        Find all words in the text where the first letter is capitalized.
        '''
        word=text.split(' ')
        word=[w for w in word if len(w)>0]
        if len([i for i in word if i[0].isupper()])==len(word):
            return [word,[1 for i in word]]
        re=[]
        i=0
        isup=[]
        for w in word:
            if w[0].isupper():
                isup.append(1)
            else:
                isup.append(0)
        return [word,isup]

    def islocation(self,locationname,country_code):
        '''
        locationname: the word want to determine if it is a location name.
        country_code: country code from the geonames data.
        return the location name, if found
        '''
        #return []
        if country_code is not None:
            df=self.location_data.loc[self.location_data['country code']==country_code].loc[self.location_data['feature class']=='A']
        else:
            df=self.location_data.loc[self.location_data['feature class']=='A']
        temp=df.loc[df['asciiname']==locationname]
        df['sim']=df['name'].map(lambda x :similar(locationname, x))
        if df['sim'].max()>0.5:
            return df.loc[df['name'].str.find(locationname)>=0].loc[df['sim']>0.5].sort_values('modification date',ascending=False)
        else:
            return []
        
        def similar(text1,text2):
            text1=text1.split(' ')
            text2=Counter(text2.split(' '))
            find_words=0
            for word1 in text1:
                if len(word1)>3:
                    if word1 in text2.keys() and text2[word1]>0:
                        text2[word1]-=1
                        find_words+=1
            return find_words/len(text2.keys())

    def getcountry_code(self,text):
        '''
        Get the country that appears in the text.
        '''
        location=text.split(' ')
        c_dict={}
        for i in location:
            tem=self.country2code
            tem['sim']=tem['Country'].map(lambda x :difflib.SequenceMatcher(None, x, i).quick_ratio())
            if tem['sim'].max()>0.9:
                c_dict[tem.sort_values('sim',ascending=False).iloc[0]['Country'],tem.sort_values('sim',ascending=False).iloc[0]['ISO']]=tem['sim'].max()
            #if len(self.country2code.loc[self.country2code['Country'].str.find(i[6:])==0])==1:
                #if i in self.country2code.loc[self.country2code['Country'].str.find(i[:-2])==0].iloc[0]['Country'] or self.country2code.loc[self.country2code['Country'].str.find(i[:-2])==0].iloc[0]['Country'] in i:
                #return tem.sort_values('sim',ascending=False).iloc[0]['Country'],tem.sort_values('sim',ascending=False).iloc[0]['ISO']#.loc[self.country2code['Country'].str.find(i[:-2])==0].values[0]
        if len(c_dict)>0:
            max_name=None
            Max_nub=0
            for i in c_dict.keys():
                if c_dict[i]>Max_nub:
                    max_name=i
            return i
        else:
            return None,None

    def getlocation(self,text):
        '''
        Get the the information which appear in the text.
        '''
        text=text.replace(',','')
        text=text.replace('.','')
        cname,ccode=self.getcountry_code(text)
        word,isup=self.getlocationword(text)
        for i,u in zip(word,isup):
            if u==1:
                if i != cname:
                    tem =self.islocation(i,ccode)
                    if len(tem)>0:# is not None:
                        return tem['name'].values[0],tem['latitude'].values[0],tem['longitude'].values[0],tem['country code'].values[0],tem['nameasc'].values[0]
        for i in range(len(word)-1):
            if isup[i] == 1 and isup[i+1] == 1:
                tem =self.islocation(word[i]+' '+word[i+1],ccode)
                if len(tem)>0:#if tem is not None:
                        return tem['name'].values[0],tem['latitude'].values[0],tem['longitude'].values[0],tem['country code'].values[0],tem['nameasc'].values[0]
        for i in range(len(word)-2):
            if isup[i] == 1 and isup[i+1] == 1 and isup[i+2] == 1:
                tem =self.islocation(word[i]+' '+word[i+1]+' '+word[i+2],ccode)
                if len(tem)>0:#if tem is not None:
                        return tem['name'].values[0],tem['latitude'].values[0],tem['longitude'].values[0],tem['country code'].values[0],tem['nameasc'].values[0]
        return cname,None,None,cname,None