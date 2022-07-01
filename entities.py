from quickumls import QuickUMLS
import json
import csv
import os

matcher = QuickUMLS(quickumls_fp = '/home/felix/quickumls')

# Folder Path
path1 = "/home/felix/article_txt_files"
path2 = "/home/felix/textbook_txt_files"
  
# Change the directory
os.chdir(path1) 
  
# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path1}/{file}"
        print(file_path)

        # call read text file function
        with open(file_path, 'r') as f:
            content = f.read()
            result = matcher.match(content, best_match=True, ignore_syntax=False)

            # write to json NOT WORK YET
            
            # Serializing json 
            json_object = json.dumps(list(result))
  
            # Writing to sample.json
            with open("entities.json", "w") as outfile:
                outfile.write(json_object)

            # write to csv
            '''''
            with open('./match_result.csv', 'a') as f:
                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                writer.writerow(result)
            '''''