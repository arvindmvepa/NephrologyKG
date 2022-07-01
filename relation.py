import json
import urllib3
import os
import csv

BASE_URL = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/"
API_KEY = "1a313e43-cbac-4194-87d3-2b2b43e63eb9"  

def get_relations(CUI, info = "partial", write_to_file = False):
    # get all relations related to the entity with given CUI

    if (info == "partial"):
        api_url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/" + CUI + "/relations?apiKey=" + API_KEY + "&sabs=MTH"
    if (info == "full"):
        api_url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/" + CUI + "/relations?apiKey=" + API_KEY

    http = urllib3.PoolManager()
    response = http.request('GET', api_url)
    json_response = json.loads(response.data)
    
    if (write_to_file):
        # Serializing json 
        json_object = json.dumps(json_response, indent=4)
  
        # Writing to sample.json
        with open("relations_" + CUI + ".json", "w") as outfile:
            outfile.write(json_object)

    return json_response

def get_entity(CUI, write_to_file = False):
    # get information about an entity using CUI

    api_url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/" + CUI + "?apiKey=" + API_KEY

    http = urllib3.PoolManager()
    response = http.request('GET', api_url)
    json_response = json.loads(response.data)

    if (write_to_file):
        json_object = json.dumps(json_response, indent=4)
  
        # Writing to sample.json
        with open("relations_" + CUI + ".json", "w") as outfile:
            outfile.write(json_object)
    
    return json_response

def get_all_relations(CUI_list, info = "partial"):
    # take a list of CUI and find all possible relations between any two CUI

    # use "for index in range(len(CUI_list)-1):" in case of checking all combinations
    for index in range(len(CUI_list)):
        json_response = get_relations(CUI_list[index], info = info)
        if "result" in json_response:
            for rel in json_response["result"]:
                #--------------Quicker Method----------------------
                if (info == "partial"):
                    url_split = rel["relatedId"].split('/')
                    related_list = [url_split[-1]]
                #--------------Slower Method-----------------------
                '''''
                elif (info == "full"):
                    http = urllib3.PoolManager()
                    api_url = rel["relatedId"] + "?apiKey=" + API_KEY
                    response = http.request('GET', api_url)
                    first_json = json.loads(response.data)

                    if "concepts" in first_json["result"]:
                        api_url = first_json["result"]["concepts"] + "&apiKey=" + API_KEY
                        response = http.request('GET', api_url)
                        second_json = json.loads(response.data)
                        related_list = []
                        for related in second_json["result"]["results"]:
                            related_list.append(related["ui"])
                        print("searching relations ...")
                    else:
                        related_list = []
                '''''
                #--------------------------------------------------

                if (len(related_list) != 0):
                    # all permutations
                    for i in range(len(CUI_list)):
                        for j in range(len(related_list)):
                            if (CUI_list[i] == related_list[j]):
                                info_for_save = {
                                    "head" : CUI_list[index], 
                                    "tail" : related_list[j], 
                                    # "ui" : rel["ui"], 
                                    "relationLabel" : rel["relationLabel"], 
                                    "additionalRelationLabel" : rel["additionalRelationLabel"], 
                                    "source" : 0
                                }
                                json_object = json.dumps(info_for_save)
                                with open("all_rel.jsonl", "a") as outfile:
                                    print(json_object, file = outfile)
                

                    # all combinations
                    '''''
                    for i in range(index+1, len(CUI_list)):
                        for j in range(len(related_list)):
                            if (CUI_list[i] == related_list[j]):
                                info_for_save = {
                                    "head" : CUI_list[index], 
                                    "tail" : related_list[j], 
                                    "ui" : rel["ui"], 
                                    "relationLabel" : rel["relationLabel"], 
                                    "additionalRelationLabel" : rel["additionalRelationLabel"]
                                }
                                json_object = json.dumps(info_for_save, indent=4)
                                with open("all_rel.json", "a") as outfile:
                                    outfile.write(json_object)
                    '''''
    return



# example
'''''
test_CUI1 = ["C0450127", "C0040739", "C1546537", "C0035015", "C2700401", "C1548437"]
test_CUI2 = ["C1843274", "C0027707", "C0041349"]
test_CUI3 = ["C0444628", "C1522486", "C1300196", "C1552679", "C0029246", "C0029237", "C2004491", "C0241158", "C1419736"]
get_all_relations(test_CUI3)
'''''

# real applications

# change to where CUI_lists.csv is stored
os.chdir("/home/felix/pyfiles")
with open('CUI_lists.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    CUI_list = list(csv_reader)

# apply get_relations
i = 0
list_len = len(CUI_list)
for list in CUI_list:
    i = i + 1
    get_all_relations(list)
    print(i, "out of", list_len, "finished")
    
