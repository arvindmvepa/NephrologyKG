import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
import joblib
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

cpnet, cpnet_simple = None, None

def generate_adj_data_for_model(data_root, sections=('dev', 'test', 'train'), k=3, add_blank=True, use_torch=True):
    nephqa_root = os.path.join(data_root, "nephqa")
    db_root = os.path.join(data_root, "ddb")

    #Convert UMLS entity linking to DDB entity linking (our KG)
    umls_to_ddb = {}
    with open(os.path.join(db_root, "ddb_to_umls_cui.txt"), encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            elms = line.split("\t")
            umls_to_ddb[elms[2]] = elms[1]

    def map_to_ddb(ent_obj):
        res = []
        for ent_cand in ent_obj['linking_results']:
            CUI  = ent_cand['Concept ID']
            name = ent_cand['Canonical Name']
            if CUI in umls_to_ddb:
                ddb_cid = umls_to_ddb[CUI]
                res.append((ddb_cid, name))
        return res

    def process(fname):
        with open(os.path.join(nephqa_root, "statement", f"{fname}.statement.umls_linked.jsonl"), encoding='utf-8')as fin:
            stmts = [json.loads(line) for line in fin]
        with open(os.path.join(nephqa_root, "grounded", f"{fname}.grounded.jsonl"), 'w', encoding='utf-8') as fout:
            for stmt in tqdm(stmts):
                sent = stmt['question']['stem']
                qc = []
                qc_names = []
                for ent_obj in stmt['question']['stem_ents']:
                    res = map_to_ddb(ent_obj)
                    for elm in res:
                        ddb_cid, name = elm
                        qc.append(ddb_cid)
                        qc_names.append(name)
                for cid, choice in enumerate(stmt['question']['choices']):
                    ans = choice['text']
                    ac = []
                    ac_names = []
                    for ent_obj in choice['text_ents']:
                        res = map_to_ddb(ent_obj)
                        for elm in res:
                            ddb_cid, name = elm
                            ac.append(ddb_cid)
                            ac_names.append(name)
                    out = {'sent': sent, 'ans': ans, 'qc': qc, 'qc_names': qc_names, 'ac': ac, 'ac_names': ac_names}
                    print (json.dumps(out), file=fout)

    os.system(f'mkdir -p {os.path.join(nephqa_root, "grounded")}')
    for fname in sections:
        process(fname)

    def load_ddb():
        with open(os.path.join(db_root, "ddb_names.json"), encoding='utf-8') as f:
            all_names = json.load(f)
        with open(os.path.join(db_root, "ddb_relas.json"), encoding='utf-8') as f:
            all_relas = json.load(f)
        relas_lst = []
        for key, val in all_relas.items():
            relas_lst.append(val)

        ddb_ptr_to_preferred_name = {}
        ddb_ptr_to_name = defaultdict(list)
        ddb_name_to_ptr = {}
        for key, val in all_names.items():
            item_name = key
            item_ptr = val[0]
            item_preferred = val[1]
            if item_preferred == "1":
                ddb_ptr_to_preferred_name[item_ptr] = item_name
            ddb_name_to_ptr[item_name] = item_ptr
            ddb_ptr_to_name[item_ptr].append(item_name)

        return relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name

    relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name = load_ddb()

    # add arbitrary blanks for defaults if there are no question or answer concepts
    if add_blank:
        item_ptrs = list(ddb_ptr_to_preferred_name.keys())
        blank_q_item_ptr = max(item_ptrs) + 1
        blank_q_name = "    "
        ddb_name_to_ptr[blank_q_name] = blank_q_item_ptr
        ddb_ptr_to_preferred_name[blank_q_item_ptr] = blank_q_name
        ddb_ptr_to_name[blank_q_item_ptr].append(blank_q_name)

        blank_a_item_ptr = blank_q_item_ptr + 1
        blank_a_name = "     "
        ddb_name_to_ptr[blank_a_name] = blank_a_item_ptr
        ddb_ptr_to_preferred_name[blank_a_item_ptr] = blank_a_name
        ddb_ptr_to_name[blank_a_item_ptr].append(blank_a_name)

    ddb_ptr_lst, ddb_names_lst = [], []
    for key, val in ddb_ptr_to_preferred_name.items():
        ddb_ptr_lst.append(key)
        ddb_names_lst.append(val)

    with open(os.path.join(db_root, "vocab.txt"), "w", encoding='utf-8') as fout:
        for ddb_name in ddb_names_lst:
            print(ddb_name, file=fout)

    with open(os.path.join(db_root, "ptrs.txt"), "w", encoding='utf-8') as fout:
        for ddb_ptr in ddb_ptr_lst:
            print(ddb_ptr, file=fout)

    id2concept = ddb_ptr_lst

    print(f"len(ddb_ptr_to_name): {len(ddb_ptr_to_name)}, "
          f"len(ddb_ptr_to_preferred_name): {len(ddb_ptr_to_preferred_name)}, "
          f"len(ddb_name_to_ptr): {len(ddb_name_to_ptr)}, len(relas_lst): {len(relas_lst)}")

    rel_types = set(rel[2] for rel in relas_lst)
    print(f"rel_types: {rel_types}")

    merged_relations = [
        'RO',
        'RN',
        'RB'
    ]

    def construct_graph():
        global concept2id, id2relation
        concept2id = {w: i for i, w in enumerate(id2concept)}
        id2relation = merged_relations
        relation2id = {r: i for i, r in enumerate(id2relation)}
        graph = nx.MultiDiGraph()
        attrs = set()
        for relation in tqdm(relas_lst):
            subj = concept2id[relation[0]]
            obj = concept2id[relation[1]]
            rel = relation2id[relation[2]]
            weight = 1.
            graph.add_edge(subj, obj, rel=rel, weight=weight)
            attrs.add((subj, obj, rel))
            graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
            attrs.add((obj, subj, rel + len(relation2id)))
        output_path = os.path.join(db_root, "ddb.graph")
        nx.write_gpickle(graph, output_path)
        return graph

    KG = construct_graph()
    load_kg(KG)

    os.system(f'mkdir -p {os.path.join(nephqa_root, "graph")}')

    for fname in sections:
        grounded_path = os.path.join(nephqa_root, "grounded", f"{fname}.grounded.jsonl")
        output_path = os.path.join(nephqa_root, "graph", "{fname}.graph.adj.pk")

        if add_blank:
            res = generate_adj_data_from_grounded_concepts(grounded_path, k, 10, blank_q_item_ptr=blank_q_item_ptr,
                                                           blank_a_item_ptr=blank_a_item_ptr)
        else:
            res = generate_adj_data_from_grounded_concepts(grounded_path, k, 10)
        print(f"Verification Sum (sum of all concept values): {sum([sum(r['concepts']) for r in res])}")

        with open(output_path, 'wb') as fout:
            joblib.dump(res, fout)

    generate_kg_embeddings(db_root, use_torch=use_torch)


def load_kg(KG):
    global cpnet, cpnet_simple
    cpnet = KG
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def generate_adj_data_from_grounded_concepts(grounded_path, k, num_processes, blank_q_item_ptr=None,
                                             blank_a_item_ptr=None):
    global concept2id

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(concept2id[int(c)] for c in dic['qc'])
            if not q_ids:
                if blank_q_item_ptr:
                    q_ids = {concept2id[blank_q_item_ptr]}
                else:
                    # arbitrary item ptr - probably not a good default
                    q_ids = {concept2id[119]}
            a_ids = set(concept2id[int(c)] for c in dic['ac'])
            if not a_ids:
                if blank_a_item_ptr:
                    a_ids = {concept2id[blank_a_item_ptr]}
                else:
                    # arbitrary item ptr - probably not a good default
                    a_ids = {concept2id[113]}
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

    if k == 2:
        concepts_to_adj_matrices_func = concepts_to_adj_matrices_2hop_all_pair
    elif k == 3:
        concepts_to_adj_matrices_func = concepts_to_adj_matrices_3hop_all_pair
    elif k == 4:
        concepts_to_adj_matrices_func = concepts_to_adj_matrices_4hop_all_pair
    else:
        raise ValueError(f"No concepts_to_adj_matrices_func for k={k}")
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_func, qa_data), total=len(qa_data)))

    lens = [len(e['concepts']) for e in res]
    print('mean #nodes', int(np.mean(lens)), 'med', int(np.median(lens)), '5th', int(np.percentile(lens, 5)),
          '95th', int(np.percentile(lens, 95)))

    return res


def concepts2adj(schema_graph, qc_ids, ac_ids, extra_nodes):
    global id2relation, cpnet
    cids = np.array(schema_graph, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if (s_c in qc_ids and t_c in ac_ids) or \
                    (s_c in qc_ids and t_c in extra_nodes) or \
                    (s_c in extra_nodes and t_c in ac_ids):
                if cpnet.has_edge(s_c, t_c):
                    for e_attr in cpnet[s_c][t_c].values():
                        if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                            adj[e_attr['rel']][s][t] = 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


def concepts2adj_for_k_gt_2(schema_graph, qc_ids, ac_ids, extra_nodes):
    global id2relation, cpnet
    cids = np.array(schema_graph, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if (s_c in qc_ids and t_c in ac_ids) or \
                    (s_c in qc_ids and t_c in extra_nodes) or \
                    (s_c in extra_nodes and t_c in ac_ids) or \
                    (s_c in extra_nodes and t_c in extra_nodes):
                if cpnet.has_edge(s_c, t_c):
                    for e_attr in cpnet[s_c][t_c].values():
                        if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                            adj[e_attr['rel']][s][t] = 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


def concepts_to_adj_matrices_2hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph, sorted(qc_ids), sorted(ac_ids), sorted(extra_nodes))
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': None}


def concepts_to_adj_matrices_3hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                # 1-hop nodes
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
                # 2-hop nodes
                for node in cpnet_simple[qid]:
                    twohop_nodes = set(cpnet_simple[node]) & set(cpnet_simple[aid])
                    if twohop_nodes:
                        # add in the one hop intermediate node
                        extra_nodes |= {node} | twohop_nodes
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj_for_k_gt_2(schema_graph, sorted(qc_ids),
                                            sorted(ac_ids), sorted(extra_nodes))
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': None}


def concepts_to_adj_matrices_4hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                # 1-hop nodes
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
                # 2-hop nodes
                for node in cpnet_simple[qid]:
                    twohop_nodes = set(cpnet_simple[node]) & set(cpnet_simple[aid])
                    if twohop_nodes:
                        # add in the one hop intermediate node
                        extra_nodes |= {node} | twohop_nodes
                    # 3-hop nodes
                    for node_ in cpnet_simple[node]:
                        threehop_nodes = set(cpnet_simple[node_]) & set(cpnet_simple[aid])
                        if twohop_nodes:
                            # add in the one hop intermediate node
                            extra_nodes |= {node_} | threehop_nodes
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj_for_k_gt_2(schema_graph, sorted(qc_ids),
                                            sorted(ac_ids), sorted(extra_nodes))
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': None}

def generate_kg_embeddings(db_root, use_torch=True):
    if use_torch:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    bert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    bert_model.to(device)
    bert_model.eval()

    with open(f"{db_root}/vocab.txt", encoding='utf-8') as f:
        names = [line.strip() for line in f]

    tensors = tokenizer(names, padding=True, truncation=True, return_tensors="pt").to(device)

    embs = []
    with torch.no_grad():
        for i, j in enumerate(tqdm(names)):
            outputs = bert_model(input_ids=tensors["input_ids"][i:i + 1],
                                 attention_mask=tensors['attention_mask'][i:i + 1])
            out = np.array(outputs[1].squeeze().tolist()).reshape((1, -1))
            embs.append(out)
    embs = np.concatenate(embs)
    np.save(f"{db_root}/ent_emb.npy", embs)


