import re
import pandas as pd
from pprint import pprint


# Gensim
import gensim
import spacy
import logging
import warnings
import gensim.corpora as corpora #gensim==4.1.2
from gensim.utils import simple_preprocess

# NLTK Stop words
from nltk.corpus import stopwords #nltk==3.6.5

import pickle




stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


global topics_


warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  


pkl_file = open('df.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close()


             
            
df=df.dropna(how='any')    #to drop if any value in the row has a nan
df = df.drop_duplicates(subset = ['FullContent'], keep = 'last').reset_index(drop = True)

#df=pd.DataFrame()
#df= pd.read_sql("select distinct cast(date as date) publish_date, Title content,CleansedContent from [UnstructuredData].dbo.dashboarddata where date >= '2021-11-10'", con=engine)
print(df.shape)  #> (2361, 3)


# Convert to list
data = df.FullContent.values.tolist()
data = set(data)


texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data]
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
texts_out=[]
for sent in texts:
    doc = nlp(" ".join(sent)) 
    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

data_ready = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    




# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]
if len(corpus)>5000:
    topics_=10
elif len(corpus)>100:
    topics_=5
else:
    topics_=0

# Build LDA model
if topics_:
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=topics_, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=10,
                                               passes=10,
                                               alpha='symmetric',
                                               iterations=100,
                                               per_word_topics=True)
    
    pprint(lda_model.print_topics()) 
    results=lda_model.print_topics() 
    results=[x[1] for x in results] 
    results=[x.split('*"')[1:] for x in results] 
    results=[[y.split('"')[0] for y in x] for x in results] 
    
    df['Topic']=''
    df['MatchProb']=''
    # now, for every doc...
    for doc_id, doc_bow in enumerate(corpus): 
        # ...get its topics...
        doc_topics = lda_model.get_document_topics(doc_bow)
        # ...& for each of its topics...
        Pmax=[x[1] for x in doc_topics]
        Pmax=Pmax.index(max(Pmax))
        
        df['Topic'][doc_id]=results[Pmax]
        df['MatchProb'][doc_id]=doc_topics[Pmax][1]
        
    df['TotalNews']=len(corpus)
        



    






