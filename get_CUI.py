import os
import json
import csv

file_path = "/home/felix/pyfiles/entities.jsonl"
# change to directory
os.chdir("/home/felix/pyfiles")

# read in the jsonl
with open(file_path, 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    cur_list = []
    for ent in result["entities"]:
        for link in ent["linking_results"]:
            if "Concept ID" in link:
                cur_list.append(link["Concept ID"])
    if not cur_list:
        continue
    with open('CUI_lists.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow(cur_list)