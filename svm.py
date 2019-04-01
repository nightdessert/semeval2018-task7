from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import tools
import numpy as np



def load_data():
    task=['1.1','1.2']
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


def main():
    parsed_texts_train,parsed_entities_train,parsed_relations_train,parsed_texts_test,parsed_entities_test,parsed_relations_test=load_data()
    print(parsed_entities_train)



main()
