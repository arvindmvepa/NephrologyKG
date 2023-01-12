import os
import json
import time
from collections import defaultdict
from subgraph import get_k_subgraph_from_db
import numpy as np


def generate_khop_kg_from_db(data_root, sections=('dev', 'test', 'train'), k=2, return_stats=True):
    kg_subgraphs = []
    for section in sections:
        print(f"Start {k}hop for {section} in {data_root}")
        nephqa_root = f'{data_root}/nephqa'
        linked_q_file_path = f'{nephqa_root}/statement/{section}.statement.umls_linked.jsonl'
        start = time.time()
        subgraphs = get_k_subgraph_from_db(linked_q_file_path, k=k)
        end = time.time()
        print(f"Generated Subgraphs. Time elapsed (s): {(end - start)}")
        kg_subgraphs = kg_subgraphs + subgraphs
    if return_stats:
        print_subgraph_stats(kg_subgraphs, k=k)

    print("Saving KG")
    db_dir = f"{data_root}/ddb"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    save_entities_file = os.path.join(db_dir, "ddb_names.json")
    save_relations_file = os.path.join(db_dir, "ddb_relas.json")
    save_ddb_to_umls_cui_file = os.path.join(db_dir, "ddb_to_umls_cui.txt")

    db_entities_json = {}
    db_to_umls = set()
    umls_to_db = dict()
    db_entity_ids = defaultdict(lambda: len(db_entity_ids))

    db_relations_json = {}
    db_relation_ids = defaultdict(lambda: len(db_relation_ids))

    for graph in kg_subgraphs:
        for paths in graph:
            if len(paths) == k:
                for path in paths:
                    entity_cuis = path[0], path[2]
                    entity_names = path[1], path[3]
                    rel = path[4]
                    # add entities
                    for cui, name in zip(entity_cuis, entity_names):
                        db_id = db_entity_ids[cui]
                        db_to_umls.add((db_id, cui))
                        umls_to_db[cui] = db_id
                        while name in db_entities_json:
                            name = name + " "
                        db_entities_json[name] = [db_id, "1"]
                    # add relations
                    subj, obj = entity_cuis
                    subj_id = db_entity_ids[subj]
                    obj_id = db_entity_ids[obj]
                    add_relation = (subj_id, obj_id, rel)
                    relation_id = db_relation_ids[add_relation]
                    db_relations_json[relation_id] = list(add_relation)

    with open(save_ddb_to_umls_cui_file, 'w', encoding='utf-8') as f:
        f.write('\t'.join(["LinkItemsToUMLSCUIID", "ItemPTR", "CUI", "ItemToUMLSCUILinkTypePTR"]) + '\n')
        for row in sorted(list(db_to_umls)):
            db_ptr = row[0]
            cui = row[1]
            row = ["0", str(db_ptr), cui, "0"]
            f.write('\t'.join(row) + '\n')

    with open(save_entities_file, 'w') as f:
        json.dump(db_entities_json, f)

    with open(save_relations_file, 'w') as f:
        json.dump(db_relations_json, f)


def merge_kg(first_db, second_db, target_db):
    if not os.path.exists(target_db):
        os.makedirs(target_db)

    first_db_entities_file = os.path.join(first_db, "ddb_names.json")
    first_db_relations_file = os.path.join(first_db, "ddb_relas.json")
    first_db_ddb_to_umls_cui_file = os.path.join(first_db, "ddb_to_umls_cui.txt")

    second_db_entities_file = os.path.join(second_db, "ddb_names.json")
    second_db_relations_file = os.path.join(second_db, "ddb_relas.json")
    second_db_ddb_to_umls_cui_file = os.path.join(second_db, "ddb_to_umls_cui.txt")

    target_db_entities_file = os.path.join(target_db, "ddb_names.json")
    target_db_relations_file = os.path.join(target_db, "ddb_relas.json")
    target_db_ddb_to_umls_cui_file = os.path.join(target_db, "ddb_to_umls_cui.txt")

    with open(first_db_entities_file) as f:
        first_db_entities_dict = json.load(f)
    with open(first_db_relations_file) as f:
        first_db_relations_dict = json.load(f)
    with open(first_db_ddb_to_umls_cui_file) as f:
        first_db_ddb_to_umls_cui_list = f.read().splitlines()[1:]
        first_db_ddb_to_umls_cui_list = [l.split("\t") for l in first_db_ddb_to_umls_cui_list]

    with open(second_db_entities_file) as f:
        second_db_entities_dict = json.load(f)
    with open(second_db_relations_file) as f:
        second_db_relations_dict = json.load(f)
    with open(second_db_ddb_to_umls_cui_file) as f:
        second_db_ddb_to_umls_cui_list = f.read().splitlines()[1:]
        second_db_ddb_to_umls_cui_list = [l.split("\t") for l in second_db_ddb_to_umls_cui_list]

    db_entities_json = {}
    db_to_umls = set()
    umls_to_db = dict()
    db_entity_ids = defaultdict(lambda: len(db_entity_ids))

    db_relations_json = {}
    db_relation_ids = defaultdict(lambda: len(db_relation_ids))

    # add entity information
    for _, str_db_id, cui, _ in first_db_ddb_to_umls_cui_list:
        db_id = db_entity_ids[cui]
        db_to_umls.add((db_id, cui))
        umls_to_db[cui] = db_id

        old_db_id = int(str_db_id)
        for name, (old_id, _) in first_db_entities_dict.items():
            if old_id == old_db_id:
                db_entities_json[name] = [db_id, "1"]

    for _, str_db_id, cui, _ in second_db_ddb_to_umls_cui_list:
        # skip cuis already added
        if cui in db_entity_ids:
            continue
        db_id = db_entity_ids[cui]
        db_to_umls.add((db_id, cui))
        umls_to_db[cui] = db_id

        old_db_id = int(str_db_id)
        for name, (old_id, _) in second_db_entities_dict.items():
            if old_id == old_db_id:
                # make sure new cuis' names don't overwrite old cui's names'
                name = name.rstrip()
                while name in db_entities_json:
                    name = name + " "
                db_entities_json[name] = [db_id, "1"]

    # add relation information
    for _, rel_tuple in first_db_relations_dict.items():
        (old_subj_id, old_obj_id, rel) = rel_tuple

        found_subj, found_obj = False, False
        for name, (old_id, _) in first_db_entities_dict.items():
            if found_subj and found_obj:
                break
            if old_id == old_subj_id:
                subj_id = db_entities_json[name][0]
                found_subj = True
            if old_id == old_obj_id:
                obj_id = db_entities_json[name][0]
                found_obj = True
        add_relation = (subj_id, obj_id, rel)
        relation_id = db_relation_ids[add_relation]
        db_relations_json[relation_id] = list(add_relation)

    for _, rel_tuple in second_db_relations_dict.items():
        (old_subj_id, old_obj_id, rel) = rel_tuple

        found_subj, found_obj = False, False
        for name, (old_id, _) in second_db_entities_dict.items():
            if found_subj and found_obj:
                break
            if old_id == old_subj_id:
                subj_id = db_entities_json[name][0]
                found_subj = True
            if old_id == old_obj_id:
                obj_id = db_entities_json[name][0]
                found_obj = True
        add_relation = (subj_id, obj_id, rel)
        # skip relations if they have already been added
        if add_relation in db_relation_ids:
            continue
        relation_id = db_relation_ids[add_relation]
        db_relations_json[relation_id] = list(add_relation)

    with open(target_db_ddb_to_umls_cui_file, 'w', encoding='utf-8') as f:
        f.write('\t'.join(["LinkItemsToUMLSCUIID", "ItemPTR", "CUI", "ItemToUMLSCUILinkTypePTR"]) + '\n')
        for row in sorted(list(db_to_umls)):
            db_ptr = row[0]
            cui = row[1]
            row = ["0", str(db_ptr), cui, "0"]
            f.write('\t'.join(row) + '\n')

    with open(target_db_entities_file, 'w') as f:
        json.dump(db_entities_json, f)

    with open(target_db_relations_file, 'w') as f:
        json.dump(db_relations_json, f)


def print_subgraph_stats(subgraphs, k=2):
    graph_counts = []
    for graph in subgraphs:
        graph_count = 0
        for path in graph:
            if len(path) == k:
                graph_count += 1
        graph_counts.append(graph_count)
    print("Subgraph Stats")
    print(f"mean = {np.mean(graph_counts)}, std = {np.std(graph_counts)}, "
          f"max = {np.max(graph_counts)}, min = {np.min(graph_counts)}")