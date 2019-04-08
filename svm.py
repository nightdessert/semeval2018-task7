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

window=2
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
            vec.append(np.random.random_sample(50,))
    return np.mean(vec,axis=0)

def load_glove():
    word_embedding={}
    f=open('glove.6B.50d.txt','r')
    for i,line in enumerate(f.readlines()):
        row=line.strip().split(' ')
        word_embedding[row[0]]=np.asarray(row[1:],dtype="float32")
    return word_embedding

def repalce_entity_text(parsed_texts,parsed_entities,):
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

def get_window_word(sentence,entity,tf_dic):
    windowl=sentence[:entity['start']].split()
    windowr=sentence[entity['end']+1:].split()
    vec=entity['text'].split()
    vec_len=(vec)
    for i in range(window):
        if len(vec)==7:break
        if len(windowl)<i+1:
            vec=['@']+vec
        else:
            vec=[windowl[i]]+vec
    for i in range(window):
        if len(vec)==7:break
        if len(windowr)<i+1:
            vec=vec+['@']
        else:
            vec+=[windowr[i]]


    result=[]
    for word in vec:
        if word in tf_dic:
            result.append(tf_dic[word])
        elif word=='@':
            result.append(-1)
        else:
            result.append(0)
    if len(result)>6: result=result[:6]
    if len(result)!=6:print(len(result))
    return np.array(result,dtype=float)


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
    entity_dic={}
    for paper in parsed_entities_train:
        for e in paper:
            entity2word_train[e['id']]=e['text']
            entity_dic[e['id']]=e
    entity2word_test={}
    for paper in parsed_entities_test:
        for e in paper:
            entity2word_test[e['id']]=e['text']
            entity_dic[e['id']]=e

    x_train=[]
    y_train=[]
    i=0
    for paper,text in zip(parsed_relations_train,parsed_texts_train):
        for e in paper:
            #paper_id=e['ent_a'].split('.')[0]


            worda=entity2word_train[e['ent_a']].split()
            window_a=np.array(get_window_word(text['text'],entity_dic[e['ent_a']],word2tfidf))
            a_len=len(worda)
            wordb=entity2word_train[e['ent_b']].split()
            window_b=np.array(get_window_word(text['text'],entity_dic[e['ent_b']],word2tfidf))
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
            #vec=np.concatenate((window_a,window_b,np.array([revers_flag])))
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

    for paper,text in zip(parsed_relations_test,parsed_texts_test):
        for e in paper:
            #paper_id=e['ent_a'].split('.')[0]
            worda=entity2word_test[e['ent_a']].split()
            window_a=np.array(get_window_word(text['text'],entity_dic[e['ent_a']],word2tfidf))
            a_len=len(worda)
            wordb=entity2word_test[e['ent_b']].split()
            window_b=np.array(get_window_word(text['text'],entity_dic[e['ent_b']],word2tfidf))
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
            #vec=np.concatenate((window_a,window_b,np.array([revers_flag])))
            if use_glove:
                vec=np.concatenate((np.array(vec),wva,wvb))
                x_test.append(vec)
            else:
                x_test.append(np.array(vec))

            y_test.append(label_list[e['type']])
    #print(x_train)
    classifier=sklearn.svm.SVC(kernel='linear')
    classifier.fit(x_train,y_train)
    predicted_svm = classifier.predict(x_test)
    print(classification_report(predicted_svm,y_test,target_names=[v for v in label_list.keys()]))




























main()
