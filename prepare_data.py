from tools import load_abstracts_relations
from collections import OrderedDict
import json
class EntityInstance(object):
    def __init__(self,item):
       
        self.question_text = item.get("text", None)
        self.start_position = item.get("start", None)
        self.end_position = item.get("end", None)
        self.qas_id = item.get("id", None)
        # print(self.start_position)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
       
        s = "{"
        s += "'question_text': %s" % (
            self.question_text)
        if not self.start_position is None:
            s += ", 'start_position': %d" % (self.start_position)
        if not self.start_position is None:
            s += ", 'end_position': %d" % (self.end_position)
        if self.qas_id:
            s += ", 'qas_id': %s" % (self.qas_id)
        s+="}"
        return s

class Instance(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 question_text,
                 context,
                 tag  ,
                 qas_id ):
        self.question_text = question_text
        self.context = context
        self.tag = tag
        self.qas_id = qas_id
        

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
       
        s = "{"
        s += "'question_text': %s" % (
            self.question_text)
        s += ", 'context':[%s]" % (self.context)
        if not self.orig_answer_text is None:
            s+= ", orig_answer_text:[%s]" % (self.orig_answer_text)
        if self.tag:
            s += ", 'tag': %s" % (self.tag)
        s+="}"
        return s

#parsed_texts:{'id':,'text'}
#parsed_entities:{'id': 'H01-1001.21', 'text': 'indices', 'start': 1290, 'end': 1297}]
#parsed_relations:{'type': 'MODEL-FEATURE', 'ent_a': 'I05-5009.13', 'ent_b': 'I05-5009.14', 'is_reverse': False}





class bert_data_handler(object):
    def __init__(self, subtask):
        self.text = OrderedDict()
        self.entities = OrderedDict()
        self.relations = []

        text_data, entities, relations = load_abstracts_relations(subtask)

        for item in text_data:

            id, text = item["id"], item["text"]
            self.text[id] = text
        for items in entities:
            for item in items:
                entity_instance = EntityInstance(item)
                self.entities[item["id"]] = entity_instance
        for items in relations:
            for item in items:
                ent_a = item["ent_a"]
                ent_b = item["ent_b"]
                assert ent_a.split(".")[0] == ent_b.split(".")[0], ('ent_a and ent_b are not in the same abstract, %s and %s'%(ent_a, ent_b))
                abstract_id = ent_a.split(".")[0]
                context = self.text[abstract_id]
                reverse = item["is_reverse"]
                if not reverse:
                    question = self.entities[ent_a].question_text +\
                                " [BREAK] " + self.entities[ent_b].question_text
                else:
                    question = self.entities[ent_b].question_text +\
                                " [BREAK] " + self.entities[ent_a].question_text
                tag = item['type']
                qas_id = ent_a + '_' + ent_b
                self.relations.append(Instance(question_text = question, 
                                               context = context,
                                               tag = tag,
                                               qas_id = qas_id))


    def dumps(self, output_path):
        return open(output_path,"w").write(json.dumps({"data":[instance.__dict__ for instance in self.relations]}))

    def loads(self, path):
      with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]




if __name__ == '__main__':
   base_path = "/home/v-yinguo/Amcute/repos/semeval2018-task7/"
   subtask = 1.2
   mode = "train"
   path = base_path + mode+ "_"+str(subtask) + ".json" 
   dh = bert_data_handler(subtask = subtask)
   dh.dumps(path)
  