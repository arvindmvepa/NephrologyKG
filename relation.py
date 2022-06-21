from gc import get_referents
import json
import os
import webbrowser

import urllib3

BASE_URL = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/"
API_KEY = "1a313e43-cbac-4194-87d3-2b2b43e63eb9"  

def get_relations(CUI, info = "partial"):
    # get all relations related to the entity with given CUI

    if (info == "partial"):
        api_url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/" + CUI + "/relations?apiKey=" + API_KEY + "&sabs=MTH"
    if (info == "full"):
        api_url = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/" + CUI + "/relations?apiKey=" + API_KEY

    http = urllib3.PoolManager()
    response = http.request('GET', api_url)
    json_response = json.loads(response.data)
    
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

    for index in range(len(CUI_list)-1):
        json_response = get_relations(CUI_list[index], info = info)
        if "result" in json_response:
            for rel in json_response["result"]:
                #--------------Quicker Method----------------------
                if (info == "partial"):
                    url_split = rel["relatedId"].split('/')
                    related_list = [url_split[-1]]
                #--------------Slower Method-----------------------
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
                    
                print(related_list)

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
    return
'''''

                for i in range(index+1, len(CUI_list)):
                    if (CUI_list[i] == related_CUI):
                        # save the current relations rel (what do we want?)
                        info_for_save = {
                            "head" : CUI_list[index], 
                            "tail" : related_CUI, 
                            "ui" : rel["ui"], 
                            "relationLabel" : rel["relationLabel"], 
                            "additionalRelationLabel" : rel["additionalRelationLabel"]
                        }
                        json_object = json.dumps(info_for_save, indent=4)
                        with open("all_rel.json", "a") as outfile:
                            outfile.write(json_object)

                        break
'''''


    

"""""
1st
"ui" =
"head" = first CUI
"tail" = second CUI
"relationlabel"
"additional ralation label"

2nd

["C0450127", "C0040739", "C1546537", "C0035015", "C2700401", "C1548437"]
"""""

# example
#get_relations("C0450127", info="full")
test_CUI = ["C0450127", "C0040739", "C1546537", "C0035015", "C2700401", "C1548437"]
get_all_relations(test_CUI, info="full")