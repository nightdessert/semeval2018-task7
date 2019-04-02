from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import sklearn
import tools
import numpy as np

transformer = TfidfTransformer(smooth_idf=False)
vectorizer = CountVectorizer(stop_words=None)
label_list={}


replace_entity_flag=False
use_glove=True


def load_data():
    task=['1.1']#,'1.2']
    #eval_list=get_eval_list()

    parsed_texts_train=[]
    parsed_entities_train=[]
    parsed_relations_train=[]
    parsed_texts_test=[]
    parsed_entities_test=[]
    parsed_relations_test=[]

    for t in task:
        parsed_texts, parsed_entities, parsed_relations=tools.load_abstracts_relations(t,False)
        parsed_texts_train+=parsed_texts
        parsed_entities_train+=parsed_entities
        parsed_relations_train+=parsed_relations
        parsed_texts, parsed_entities, parsed_relations=tools.load_abstracts_relations(t,True)
        parsed_texts_test+=parsed_texts
        parsed_entities_test+=parsed_entities
        parsed_relations_test+=parsed_relations

    return parsed_texts_train,parsed_entities_train,parsed_relations_train,parsed_texts_test,parsed_entities_test,parsed_relations_test

def start_word_pos(sentence,pos):
    return len(sentence[:pos].split())


def end_word_pos(sentence,pos):
    return len(sentence[:pos+1].split())-1

def get_tfidfvec(tf_dic,wordlist):
    vec=[]
    for word in wordlist:
        if word in tf_dic:
            vec.append(tf_dic[word])
        else:
            vec.append(0.0)
    return np.mean(vec)

def get_glove(word_embedding, wordl):
    vec=[]
    for word in wordl:
        if word in word_embedding:
            vec.append(word_embedding[word])
        else:
            vec.append(np.random.random_sample(300,))
    return np.mean(vec,axis=0)

def load_glove():
    word_embedding={}
    f=open('glove.6B.300d.txt','r')
    for i,line in enumerate(f.readlines()):
        row=line.strip().split(' ')
        word_embedding[row[0]]=np.asarray(row[1:],dtype="float32")
    return word_embedding

def repalce_entity_text(parsed_texts,parsed_entities):
    corpus=[]
    entityl={}
    pattern='entity_'
    i=0
    for text,entity_list in zip(parsed_texts,parsed_entities):
        #print(text)
        text=text['text']
        for e in entity_list:
            if e['id'] not in entityl:
                entityl[e['id']]=pattern+str(i)
                i+=1

            text=text.replace(e['text'],entityl[e['id']])
        corpus.append(text)
    return corpus,entityl
def main():
    if use_glove:
        word_embedding=load_glove()
    parsed_texts_train,parsed_entities_train,parsed_relations_train,parsed_texts_test,parsed_entities_test,parsed_relations_test=load_data()
    paperid2index_train={e['id']:index for index,e in enumerate(parsed_texts_train)}
    paperid2index_test={e['id']:index for index,e in enumerate(parsed_texts_test)}

    sentences_train=[s['text'] for s in parsed_texts_train]

    sentences_test=[s['text'] for s in parsed_texts_test]
    sentences=sentences_train+sentences_test


    if replace_entity_flag:
        sentences,entityl=repalce_entity_text(parsed_texts_train+parsed_texts_test,parsed_entities_train+parsed_entities_test)

    counts = vectorizer.fit_transform(sentences)
    #counts_test = vectorizer.fit_transform(sentences_test)
    #print(vectorizer.vocabulary_)

    tfidf = transformer.fit_transform(counts)
    word2tfidf={}
    word2tfidf = {a:b for a,b in zip(vectorizer.get_feature_names(), transformer.idf_)}
    #print(vectorizer.get_feature_names())
    entity2word_train={}
    for paper in parsed_entities_train:
        for e in paper:
            entity2word_train[e['id']]=e['text']
    entity2word_test={}
    for paper in parsed_entities_test:
        for e in paper:
            entity2word_test[e['id']]=e['text']

    x_train=[]
    y_train=[]
    i=0
    for paper in parsed_relations_train:
        for e in paper:
            #paper_id=e['ent_a'].split('.')[0]

            worda=entity2word_train[e['ent_a']].split()
            a_len=len(worda)
            wordb=entity2word_train[e['ent_b']].split()
            b_len=len(wordb)

            if use_glove:
                wva=get_glove(word_embedding,worda)
                wvb=get_glove(word_embedding,wordb)

            worda=get_tfidfvec(word2tfidf,worda)
            wordb=get_tfidfvec(word2tfidf,wordb)



            if replace_entity_flag:
                if entityl[e['ent_a']] in word2tfidf:
                    worda=word2tfidf[entityl[e['ent_a']]]
                else:
                    worda=0.0
                if entityl[e['ent_b']] in word2tfidf:
                    wordb=word2tfidf[entityl[e['ent_b']]]
                else:
                    wordb=0.0

            revers_flag=0
            if e['is_reverse']==True: revers_flag=1
            #vec=[worda,wordb,revers_flag]

            vec=[worda,a_len,wordb,b_len,revers_flag]
            if use_glove:
                vec=np.concatenate((np.array(vec),wva,wvb))
                x_train.append(vec)
            else:
                x_train.append(np.array(vec))

            if e['type'] not in label_list:
                label_list[e['type']]=i
                i+=1
            y_train.append(label_list[e['type']])

    x_test=[]
    y_test=[]

    for paper in parsed_relations_test:
        for e in paper:
            #paper_id=e['ent_a'].split('.')[0]
            worda=entity2word_test[e['ent_a']].split()
            a_len=len(worda)
            wordb=entity2word_test[e['ent_b']].split()
            b_len=len(wordb)

            if use_glove:
                wva=get_glove(word_embedding,worda)
                wvb=get_glove(word_embedding,wordb)

            worda=get_tfidfvec(word2tfidf,worda)
            wordb=get_tfidfvec(word2tfidf,wordb)

            if replace_entity_flag:
                if entityl[e['ent_a']] in word2tfidf:
                    worda=word2tfidf[entityl[e['ent_a']]]
                else:
                    worda=0.0
                if entityl[e['ent_b']] in word2tfidf:
                    wordb=word2tfidf[entityl[e['ent_b']]]
                else:
                    wordb=0.0
            revers_flag=0
            if e['is_reverse']==True: revers_flag=1
            #vec=[worda,wordb,revers_flag]
            vec=[worda,a_len,wordb,b_len,revers_flag]
            if use_glove:
                vec=np.concatenate((np.array(vec),wva,wvb))
                x_test.append(vec)
            else:
                x_test.append(np.array(vec))

            y_test.append(label_list[e['type']])
    for x in x_train:
        if x.shape[0]!=105:
            print(x.shape[0])
    classifier=sklearn.svm.SVC(kernel='rbf')
    classifier.fit(x_train,y_train)
    predicted_svm = classifier.predict(x_test)
    print(classification_report(predicted_svm,y_test,target_names=[v for v in label_list.keys()]))




























main()
