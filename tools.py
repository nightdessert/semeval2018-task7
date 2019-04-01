



import os
import re
import shlex
import typing
import tempfile
import itertools
import subprocess
import collections

# installed modules
import spacy
import numpy as np
from bs4 import BeautifulSoup


INVALID_TAGS = ['text', 'title', 'abstract', 'body', 'html']
def load_abstracts_relations(
        subtask: typing.Union[float, int, str],
        load_test: bool=False
) -> typing.Tuple[list, list, list]:
    """
    Load abstracts, entities, and relations from dataset.

    :param subtask: Subtask to consider. Choose from 1.1, 1.2, or 2.
    :type subtask: typing.Union[float, int, str]

    :return: a tuple cointains:
        -   parsed_texts: a list of dictionaries, each of which containing
            the following keyed values:
                +   id: identifier for the document
                +   text: text of the doucment
        -   parsed_entities: a list of lists of entities in each
            document. Each list of entities associated with a document
            contains the following keyed values:
                +   id: entity identifier
                +   text: textual represenation of the entity
        -   parsed_relations: a list of lists of relations in each
            document. Each list of relations associated with a document
            contains the following keyed values:
                +   ent_a: entity involved in the relationship
                +   ent_b: entity involved in the relationship
                +   is_reverse: if false, relation is ent_a -> ent_b;
                    if true, relation is ent_b -> ent_a
                +   type: type of relationship
    :rtype: typing.Tuple[list, list, list]
    """

    if not load_test:
        # path to dataset containing abstracts and entities
        dataset_path = os.path.join('training-data', str(subtask),'text.xml')
    else:
        dataset_path = os.path.join('test-data', str(subtask), 'text.xml')

    with open(dataset_path) as f:
        soup = BeautifulSoup(f, 'xml')

    parsed_entities = []
    parsed_texts = []

    # iterate over all titles/abstracts in the input text
    for title_abstract in soup.find_all('text'):

        text_id = title_abstract.attrs['id']

        parsed_passage = ''

        # add a full stop after the title of the paper,
        # so it can be merged with the abstract.
        title_abstract = BeautifulSoup(
            str(title_abstract).replace('</title>', '</title>.'),
            'lxml'
        )

        # remove tags arount title/abstract that we don't care about
        for tag in INVALID_TAGS:
            for match in title_abstract.findAll(tag):
                match.unwrap()

        # throw away all xml syntax, only keep the raw info for
        # easy parsing and removal of entities. we want to go from
        #   ... rhetorical and <entity id="L08-1459.34">syntactic</entity>
        #   <entity id="L08-1459.35">properties</entity> of parentheticals
        #   as well as the <entity id="L08-1459.36">decisions</entity> ...
        # to
        #   ... rhetorical and syntactic properties of parentheticals
        #   as well as the decisions ...
        # plus a list of entities.
        title_abstract = str(title_abstract)

        # remove unwanted spaces and add spaces when not set.
        title_abstract = re.sub(r'<entity ', r' <entity ', title_abstract)
        title_abstract = re.sub(r'>\s*', r'>', title_abstract)
        title_abstract = re.sub(r'\s*</entity>', r'</entity> ', title_abstract)
        title_abstract = re.sub(r'\s+', ' ', title_abstract).strip()
        parsed_entities.append([])

        i = 0   # moving pointer to character in the passsage
        while True:
            # keep looping until all entities have been extracted

            # find the start of the next entity
            next_entity = title_abstract.find('<entity', i)

            if next_entity < 0:
                # if the start is -1, there are no more entities to
                # extract, so add the last remaining bit of text to
                # parsed passage and get out of the while True loop.
                parsed_passage += title_abstract[i:]
                break

            # add the text between the last entity extacted and
            # the new entity to the parsed passage.
            parsed_passage += title_abstract[i:next_entity]

            # determine where the actual entity starts and ends, ignoring
            # the "<entity>" and "</entity>" tags.
            start_entity = title_abstract.find('">', next_entity) + 2
            end_entity = title_abstract.find('</entity>', start_entity)

            # extract the text of the entity
            entity_text = title_abstract[start_entity:end_entity]

            # extract the id of the entity
            entity_id = re.search(
                "\"(.+?)\"", title_abstract[next_entity:start_entity]
            ).groups()[0]

            # calculate where to start to scan next based on the
            # end entity offset and the length on the tag
            i = end_entity + len('</entity>')

            # derive position of the entity in the new parsed passage
            start_entity_in_parsed_passage = len(parsed_passage)
            parsed_passage += entity_text
            end_entity_in_parsed_passage = len(parsed_passage)

            # add entity to list of entitities
            parsed_entities[-1].append({
                'id': entity_id,
                'text': entity_text,
                'start': start_entity_in_parsed_passage,
                'end': end_entity_in_parsed_passage
            })

        # add parsed text to list of texts.
        parsed_texts.append({
            'id': text_id,
            'text': parsed_passage
        })

    if not load_test:
        # path to file containing relations
        relations_path = os.path.join(
            'training-data', str(subtask), 'relations.txt'
        )
    else:
        # path to file containing relations
        relations_path = os.path.join(
            'test-data', str(subtask), 'relations.txt'
        )

    # set up list to group relations by document
    docs_ids = {text['id']: i for i, text in enumerate(parsed_texts)}
    parsed_relations = [[] for _ in docs_ids]


    # get relations out
    # note that no relations are given for subtask 2 test set
    with open(relations_path) as f:
        for ln in f:

            # strip end characters
            ln = ln.strip()

            # skip empty lines
            if not ln:
                continue

            # data format:
            # RELATION_TYPE(<ENTITY>,<ENTITY>)
            # or
            # RELATION_TYPE(<ENTITY>,<ENTITY>,<REVERSE>)

            # if not load_test:
            # separate relation type from entities
            rel_type, rel_data = ln.strip(')').split('(')
            # else:
            # rel_type = None
            # rel_data = ln.strip(')').strip('(')

            # parse entities, reverse if avaliable
            try:
                ent_a, ent_b, is_reverse = rel_data.split(',')
            except ValueError:
                ent_a, ent_b = rel_data.split(',')
                is_reverse = False

            # use doc id to determine the position in parsed_relations list
            doc_id = ent_a.split('.')[0]

            # casting to prevent warning in PyCharm
            doc_pos = int(docs_ids[doc_id])

            parsed_relations[doc_pos].append({
                'type': rel_type,
                'ent_a': ent_a,
                'ent_b': ent_b,
                'is_reverse': is_reverse,
            })

    return parsed_texts, parsed_entities, parsed_relations
    #parsed_texts:{'id':,'text'}
    #parsed_entities:{'id': 'H01-1001.21', 'text': 'indices', 'start': 1290, 'end': 1297}]
    #parsed_relations:{'type': 'MODEL-FEATURE', 'ent_a': 'I05-5009.13', 'ent_b': 'I05-5009.14', 'is_reverse': False}



def get_eval_list():
    eval_list=[]
    for l in open('data/training-eval.txt'):
        l=l.strip('\n').split(' ')
        if l[0]=='2': break
        eval_list.append(l[1])
    return eval_list
