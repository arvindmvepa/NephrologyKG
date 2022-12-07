import json
import asyncio
import aiohttp
from relation import async_get_relations
from db import connect_db


def get_concepts_from_questions(linked_question_file):
    # load linked question file
    with open(linked_question_file) as f:
        lines = f.readlines()
    json_lines = [json.loads(line) for line in lines]

    # get all the concepts per question text
    question_cui_name_pairs = []
    for json_line in json_lines:
        ent_results = [v for ent in json_line['question']['stem_ents']
                       for k, v in ent.items() if k == 'linking_results']
        cui_name_pairs = [(ent['Concept ID'], ent['Canonical Name']) for ent_matches in ent_results
                for ent in ent_matches]
        question_cui_name_pairs.append(cui_name_pairs)

    # get all the concepts per choice per question
    answer_cui_name_pairs = []
    for json_line in json_lines:
        choice_cuis = []
        for choice in json_line['question']['choices']:
            cuis = [(ent_match['Concept ID'], ent_match['Canonical Name'])
                    for ent_results in choice['text_ents']
                    for ent_match in ent_results['linking_results']]
            choice_cuis.append(cuis)
        answer_cui_name_pairs.append(choice_cuis)

    return question_cui_name_pairs, answer_cui_name_pairs


async def get_all_related_cuis_names(cui, session):
    print(f"Getting relations for: {cui}")
    json_response = await async_get_relations(cui, session)
    all_related_cuis_names = []
    if "result" in json_response:
        for rel in json_response["result"]:
            url_split = rel["relatedId"].split('/')
            related_cuis = [url_split[-1]]
            if "relatedIdName" not in rel:
                continue
            related_names = [rel["relatedIdName"]]
            if len(related_cuis) > 0:
                all_related_cuis_names.extend([(rel_cui, rel_name, rel["relationLabel"], rel["additionalRelationLabel"])
                                         for rel_cui, rel_name in zip(related_cuis, related_names)])
    print(f"Finished getting relations for: {cui}")
    return all_related_cuis_names


async def get_one_hop_paths(source_cui_name_pairs, dest_cui_name_pairs, session, index):
    one_hop_paths = []
    dest_cuis = [cui for (cui, name_pair) in dest_cui_name_pairs]
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        #print(f"{index}  {i}")
        target_cuis_names_rels = await get_all_related_cuis_names(source_cui, session)
        one_hop_paths.extend([[(source_cui, source_name, target_cui, target_name, target_rel1, target_rel2)]
                              for target_cui, target_name, target_rel1, target_rel2 in target_cuis_names_rels
                              if target_cui in dest_cuis])
    return one_hop_paths


async def get_two_hop_paths(source_cui_name_pairs, dest_cui_name_pairs, session, index):
    two_hop_paths = []
    dest_cuis = [cui for (cui, name_pair) in dest_cui_name_pairs]
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        #print(f"{index}  {i}")
        int_cuis_names_rels = await get_all_related_cuis_names(source_cui, session)
        for j, (int_cui, int_name, int_rel1, int_rel2) in enumerate(int_cuis_names_rels):
            #print(f"{index}  {i}.{j}")
            target_cuis_names_rels = await get_all_related_cuis_names(int_cui, session)
            two_hop_paths.extend([[(source_cui, source_name, int_cui, int_name, int_rel1, int_rel2),
                                   (int_cui, int_name, target_cui, target_name, target_rel1, target_rel2)]
                                  for target_cui, target_name, target_rel1, target_rel2 in target_cuis_names_rels
                                  if target_cui in dest_cuis])
    return two_hop_paths


async def get_three_hop_paths(source_cui_name_pairs, dest_cui_name_pairs, session, index):
    three_hop_paths = []
    dest_cuis = [cui for (cui, name_pair) in dest_cui_name_pairs]
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        #print(f"{index}  {i}")
        int_cuis_names_rels1 = await get_all_related_cuis_names(source_cui, session)
        for j, (int_cui, int_name, int_rel1, int_rel2) in enumerate(int_cuis_names_rels1):
            int_cuis_names_rels2 = await get_all_related_cuis_names(int_cui, session)
            for k, (int_cui_, int_name_, int_rel1_, int_rel2_) in enumerate(int_cuis_names_rels2):
                #print(f"{index}  {i}.{j}")
                target_cuis_names_rels = await get_all_related_cuis_names(int_cui_, session)
                three_hop_paths.extend([[(source_cui, source_name, int_cui, int_name, int_rel1, int_rel2),
                                         (int_cui, int_name, int_cui_, int_name_, int_rel1_, int_rel2_),
                                         (int_cui_, int_name_, target_cui, target_name, target_rel1, target_rel2)]
                                      for target_cui, target_name, target_rel1, target_rel2 in target_cuis_names_rels
                                      if target_cui in dest_cuis])
    return three_hop_paths


async def get_four_hop_paths(source_cui_name_pairs, dest_cui_name_pairs, session, index):
    four_hop_paths = []
    dest_cuis = [cui for (cui, name_pair) in dest_cui_name_pairs]
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        #print(f"{index}  {i}")
        int_cuis_names_rels1 = await get_all_related_cuis_names(source_cui, session)
        for j, (int_cui, int_name, int_rel1, int_rel2) in enumerate(int_cuis_names_rels1):
            int_cuis_names_rels2 = await get_all_related_cuis_names(int_cui, session)
            for k, (int_cui_, int_name_, int_rel1_, int_rel2_) in enumerate(int_cuis_names_rels2):
                int_cuis_names_rels3 = await get_all_related_cuis_names(int_cui_, session)
                for l, (int_cui__, int_name__, int_rel1__, int_rel2__) in enumerate(int_cuis_names_rels3):
                    #print(f"{index}  {i}.{j}")
                    target_cuis_names_rels = await get_all_related_cuis_names(int_cui__, session)
                    four_hop_paths.extend([[(source_cui, source_name, int_cui, int_name, int_rel1, int_rel2),
                                             (int_cui, int_name, int_cui_, int_name_, int_rel1_, int_rel2_),
                                             (int_cui_, int_name_, int_cui__, int_name__, int_rel1__, int_rel2__),
                                            (int_cui__, int_name__, target_cui, target_name, target_rel1, target_rel2)]
                                           for target_cui, target_name, target_rel1, target_rel2 in target_cuis_names_rels
                                           if target_cui in dest_cuis])
    return four_hop_paths


async def get_onehop_subgraph(linked_question_file):
    """Get 2-hop subgraphs from question_cuis and answer_cuis using parallelized approach.
    For parallelizing http requests, reference this:
    https://stackoverflow.com/questions/57126286/fastest-parallel-requests-in-python
    """
    question_cui_name_pairs, answer_cui_name_pairs = get_concepts_from_questions(linked_question_file)
    question_cui_name_pairs, answer_cui_name_pairs = question_cui_name_pairs, answer_cui_name_pairs
    async with aiohttp.ClientSession() as session:
        subgraphs = await asyncio.gather(*[get_one_hop_paths(q_cui_cui_name_pair, a_choice_cui_name_pair, session, (i,j))
                                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)])
    print("Finalized all. Return is a list of len {} outputs.".format(len(subgraphs)))
    return subgraphs


async def get_twohop_subgraph(linked_question_file):
    """Get 2-hop subgraphs from question_cuis and answer_cuis using parallelized approach.
    For parallelizing http requests, reference this:
    https://stackoverflow.com/questions/57126286/fastest-parallel-requests-in-python
    """
    question_cui_name_pairs, answer_cui_name_pairs = get_concepts_from_questions(linked_question_file)
    question_cui_name_pairs, answer_cui_name_pairs = question_cui_name_pairs, answer_cui_name_pairs
    async with aiohttp.ClientSession() as session:
        subgraphs = await asyncio.gather(*[get_two_hop_paths(q_cui_cui_name_pair, a_choice_cui_name_pair, session, (i,j))
                                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)])
    print("Finalized all. Return is a list of len {} outputs.".format(len(subgraphs)))
    return subgraphs


async def get_threehop_subgraph(linked_question_file):
    """Get 2-hop subgraphs from question_cuis and answer_cuis using parallelized approach.
    For parallelizing http requests, reference this:
    https://stackoverflow.com/questions/57126286/fastest-parallel-requests-in-python
    """
    question_cui_name_pairs, answer_cui_name_pairs = get_concepts_from_questions(linked_question_file)
    question_cui_name_pairs, answer_cui_name_pairs = question_cui_name_pairs, answer_cui_name_pairs
    async with aiohttp.ClientSession() as session:
        subgraphs = await asyncio.gather(*[get_three_hop_paths(q_cui_cui_name_pair, a_choice_cui_name_pair, session, (i,j))
                                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)])
    print("Finalized all. Return is a list of len {} outputs.".format(len(subgraphs)))
    return subgraphs


async def get_fourhop_subgraph(linked_question_file):
    """Get 2-hop subgraphs from question_cuis and answer_cuis using parallelized approach.
    For parallelizing http requests, reference this:
    https://stackoverflow.com/questions/57126286/fastest-parallel-requests-in-python
    """
    question_cui_name_pairs, answer_cui_name_pairs = get_concepts_from_questions(linked_question_file)
    question_cui_name_pairs, answer_cui_name_pairs = question_cui_name_pairs[:5], answer_cui_name_pairs[:5]
    async with aiohttp.ClientSession() as session:
        subgraphs = await asyncio.gather(*[get_four_hop_paths(q_cui_cui_name_pair, a_choice_cui_name_pair, session, (i,j))
                                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)])
    print("Finalized all. Return is a list of len {} outputs.".format(len(subgraphs)))
    return subgraphs


def get_k_subgraph_from_db(linked_question_file, k=2):
    """Get k-hop subgraphs from question_cuis and answer_cuis from db"""
    db_cnx = connect_db()
    cursor = db_cnx.cursor()
    question_cui_name_pairs, answer_cui_name_pairs = get_concepts_from_questions(linked_question_file)
    question_cui_name_pairs, answer_cui_name_pairs = question_cui_name_pairs, answer_cui_name_pairs

    if k == 1:
        subgraphs = [get_one_hop_paths_from_db(q_cui_cui_name_pair, a_choice_cui_name_pair, cursor, (i,j))
                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)]
    elif k == 2:
        subgraphs = [get_two_hop_paths_from_db(q_cui_cui_name_pair, a_choice_cui_name_pair, cursor, (i,j))
                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)]
    elif k == 3:
        subgraphs = [get_three_hop_paths_from_db(q_cui_cui_name_pair, a_choice_cui_name_pair, cursor, (i,j))
                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)]
    elif k == 4:
        subgraphs = [get_four_hop_paths_from_db(q_cui_cui_name_pair, a_choice_cui_name_pair, cursor, (i,j))
                     for i, (q_cui_cui_name_pair, a_choices_cui_name_pairs) in enumerate(zip(question_cui_name_pairs, answer_cui_name_pairs))
                     for j, a_choice_cui_name_pair in enumerate(a_choices_cui_name_pairs)]
    else:
        raise ValueError(f"subgraphs for k={k} is not defined")
    cursor.close()
    db_cnx.close()
    print("Finalized all. Return is a list of len {} outputs.".format(len(subgraphs)))

    return subgraphs


def get_one_hop_paths_from_db(source_cui_name_pairs, dest_cui_name_pairs, cursor, index):
    one_hop_paths = []
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        for j, (dest_cui, dest_name) in enumerate(dest_cui_name_pairs):
            query = "SELECT t1.CUI1,t1.CUI2,t1.REL,t1.RELA FROM MRREL_MTH t1 " \
                    f"WHERE t1.CUI1='{source_cui}' and t1.CUI2='{dest_cui}' ;"
            cursor.execute(query)
            for res in cursor:
                _, _, rel, rela = res
                one_hop_paths.append([source_cui, source_name, dest_cui, dest_name, rel, rela])
    return one_hop_paths


def get_two_hop_paths_from_db(source_cui_name_pairs, dest_cui_name_pairs, cursor, index):
    two_hop_paths = []
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        for j, (dest_cui, dest_name) in enumerate(dest_cui_name_pairs):
            query = f"SELECT t1.CUI2,f1.STR,t1.REL,t1.RELA,t2.CUI1,f2.STR,t2.REL,t2.RELA FROM MRREL_MTH t1, MRREL_MTH t2, " \
                    f"MRCONSO f1, MRCONSO f2 WHERE t1.CUI1='{source_cui}' and t1.CUI2=t2.CUI1 and t2.CUI2='{dest_cui}' " \
                    "and t1.CUI2=f1.CUI and t2.CUI1=f2.CUI ;"
            cursor.execute(query)
            for res in cursor:
                cui2_1, name_1, rel_1, rela_1, cui1_2, name_2, rel_2, rela_2 = res
                two_hop_paths.append([source_cui, source_name, cui2_1, name_1, rel_1, rela_1])
                two_hop_paths.append([cui2_1, name_1, dest_cui, dest_name, rel_2, rela_2])
    return two_hop_paths


def get_three_hop_paths_from_db(source_cui_name_pairs, dest_cui_name_pairs, cursor, index):
    three_hop_paths = []
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        for j, (dest_cui, dest_name) in enumerate(dest_cui_name_pairs):
            query = f"SELECT t1.CUI2,t1.REL,t1.RELA,t2.CUI1,t2.CUI2,t2.REL,t2.RELA," \
                    "t3.CUI1,t3.REL,t3.RELA FROM MRREL_MTH t1, MRREL_MTH t2, MRREL_MTH t3, MRCONSO f1, MRCONSO f2, MRCONSO f3 " \
                    f"WHERE t1.CUI1='{source_cui}' and t1.CUI2=t2.CUI1 t2.CUI2=t3.CUI1 and t3.CUI2='{dest_cui}';"
            cursor.execute(query)
            for res in cursor:
                _, _, rel, rela = res
                three_hop_paths.append([source_cui, source_name, dest_cui, dest_name, rel, rela])
    return three_hop_paths


def get_four_hop_paths_from_db(source_cui_name_pairs, dest_cui_name_pairs, cursor, index):
    four_hop_paths = []
    for i, (source_cui,source_name) in enumerate(source_cui_name_pairs):
        for j, (dest_cui, dest_name) in enumerate(dest_cui_name_pairs):
            query = "SELECT t1.CUI1,t1.CUI2,t1.REL,t1.RELA,t2.CUI1,t2.CUI2,t2.REL,t2.RELA, " \
                    "t3.CUI1,t3.CUI2,t3.REL,t3.RELA,t4.CUI1,t4.CUI2,t4.REL,t4.RELA " \
                    "FROM MRREL_MTH t1, MRREL_MTH t2, MRREL_MTH t3, MRREL_MTH t4 " \
                    f"WHERE t1.CUI1='{source_cui}' and t1.CUI2=t2.CUI1 t2.CUI2=t3.CUI1 " \
                    f"and t3.CUI2=t4.CUI1 and t4.CUI2='{dest_cui}';"
            cursor.execute(query)
            for res in cursor:
                _, _, rel, rela = res
                four_hop_paths.append([source_cui, source_name, dest_cui, dest_name, rel, rela])
    return four_hop_paths
