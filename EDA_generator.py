from encodings import utf_8
import numpy as np
import requests
#import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
import re
import math
import os
#from sympy import li
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from operator import itemgetter
from collections import Counter



#To run
#python EDA_generator.py

#Article txt path: C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_txt_files
#Textbook txt path: C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_txt_files

#Preprocessing
###
# 1. Lower everything
# 2. Change contractions to two words: “don’t” -> “do not”
# 3. Removes all characters except A-Z, a-z, -, \n
# 4. Tokenizes by sentence (creates a list of sentences)
# 5. Dealing with the hyphen problems:
#    Want to remove all instances of -\n (ex. reno-\nvascular) but not hyphens in words like drug-induced
#    Hyphen problem 1: need to replace '-\n' with ''
#    Hyphen problem 2: need to remove the rest of the \n 
    #need to remove stopwords
# 6. Remove strange length words
#       These tended to be figure captions that melded together- 
#       figured anything that was being described in the image would also be described in the text
# EDA functions:
#   -Average sentence length
#   -Top 100 words and their counts
#   -Topic model

contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

#Run all files in file path through pdfminer
#input folder path w/ all pdfs, out folder where you want the txt files stored
def run_pdfminer(in_dir, out_dir):
    count = 0
    for filename in os.listdir(in_dir):
        file = in_dir + '\\\\' + filename
        if os.path.isfile(file):
            os.system('python pdf2txt.py ' + file + ' -o ' + out_dir + '\\\\' + filename[:-4] + '.txt -t text')
            print(count)
            count += 1

#Call to run textbookfiles through pdfminer
#run_pdfminer('C:\\\\Users\\\\kiana\\\\COSC490\\\\AI_Nephrologist\\\\textbook_pdf_files', 'C:\\\\Users\\\\kiana\\\\COSC490\\\\AI_Nephrologist\\\\textbook_txt_files')

#Contraction Removal Function
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys())) #honeslty no clue what this is
def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_re.sub(replace, s)

#Stopwords
stop_words = set(stopwords.words('english'))
data_specific_stop_words = {'patients', 'patient', 'associated', 'usually', 'however', 'often', 'recent', 'may', 'treatment', 'figure', 'fig', 'management', 'complications', 'studies', 'study', 'cases', 'case', 'time', 'without', 'one', 'see', 'control', 'significant', 'reduced', 'use', 'also', 'performed', 'used', 'many', 'might'}
stop_words.update(data_specific_stop_words)

#Stemmer- unused rn
def stem_words(text):
    return " ".join([PorterStemmer().stem(word) for word in text.split()])

#Lemmatizer
wnl = WordNetLemmatizer()

#Hyphen problem 1- remove instances of -\n
def hypen_problem_one(in_list):
    new_list = []
    for sentence in in_list:
        sentence = re.sub('-\n', '', sentence)
        new_list.append(sentence)
    return new_list

#Hyphen problem 2- remove the rest of the \n characters
def hypen_problem_two(in_list):
    new_list =[]
    for sentence in in_list:
        new_sentence = ''
        #change \n characters to spaces
        sentence = re.sub('\n', ' ', sentence)
        #change tabs to spaces
        sentence = re.sub("[^A-Za-z -]", "", sentence)
        sentence = re.sub(' +', ' ', sentence)
        tokens = word_tokenize(sentence)
        tokens = [i for i in tokens if not i in stop_words]
        for item in tokens:
            new_sentence += item + ' '
        new_list.append(new_sentence)
    return new_list

#Removing extra long "words" (melded sentences/figure captions)
def remove_strange_length_words(in_list):
    new_list = []
    for sentence in in_list:
        new_sentence = ''
        tokens = word_tokenize(sentence) #tokens = list of words in one sentence
        for word in tokens:
            if len(word) < 30 and len(word) > 2:
                new_sentence = new_sentence + word + ' '
        new_list.append(new_sentence)
    return new_list


#Preprocessor:
# takes the txt file and preprocesses it
def preprocess (in_file): #in_file is the unprocessed .txt file file path (str), returns a list of words
    # open the data file
    file = open(in_file, encoding = 'utf8') #idk what encoding is but we needed this
    # read the file 
    data = file.read() #data is a string
    #print(data)
    # close the file
    file.close()

    # 1. Lower text
    data = data.lower()

    # 2. Change contractions
    data = expand_contractions(data)
    
    #3. Removing all charcters except for A-Z a-z .  - and space
    #   Comeback to hypens issues
    data = re.sub("[^A-Za-z .r\n-]", "", data)
    data = re.sub(' +', ' ', data)

    #4. Tokenizes by sentence
    sent_data = sent_tokenize(data)
    
    # gives me list of sentences, with \n charcters and =
    # looping through list now
    # data is a list of sentences now*****

    #5. Dealing with the hyphen problems
    #Fix hypen problem 1 (see above)
    sent_data = hypen_problem_one(sent_data)
    #Fix hypen problem two (see above)
    sent_data = hypen_problem_two(sent_data)

    #6. Remove bad length words in sentences
    sent_data = remove_strange_length_words(sent_data)

    #for counter test


    # Normal return
    return sent_data



#Gives list of words- for other EDA functions
#other functions call preprocess(list_of_words(in_folder))
def list_of_words(sent_list):
    word_data = []
    for sentence in sent_list:
        tokens = word_tokenize(sentence)
        for words in tokens:
            lemma_word = wnl.lemmatize(words)
            word_data.append(lemma_word)
    return word_data


#EDA tools


#Gives average sentence length
#Takes a list of file paths
def average_sentence_length(in_folder):
    sentence_lengths = []
    for filename in os.listdir(in_folder):
        file = in_folder + '\\' + filename
        prep_sent_list = preprocess(file)
        for sentence in prep_sent_list:
            tokens = word_tokenize(sentence)
            count = len(tokens)
            sentence_lengths.append(count)
        
    sentence_total = 0
    for num in sentence_lengths:
        sentence_total += num
    sentence_avg = sentence_total / len(sentence_lengths)
    return sentence_avg

#Writes a word count dictionary
#takes a list of str file paths and an out file path- writes a word count dictionary in out path
def word_count_dictionary(in_folder, output_file):
    with open(output_file, 'a') as outfile:
        final_list = []
        count = 0
        for filename in os.listdir(in_folder):
            file = in_folder + '\\\\' + filename
            with open(file) as infile:
                final_list += list_of_words(preprocess(file))
            print(count)
            count += 1
        word_count_dict = {}
        for key in final_list:
            word_count_dict[key] = final_list.count(key)
        outfile.write(repr(word_count_dict))
    return word_count_dict

#Gives the top 100 words and their counts
#Takes a list of str file paths and an out file path- writes the top 100 used words in word count dictionary in out path
def highest_dict_value(in_folder, output_file):
    count = 0
    with open(output_file, 'a') as outfile:
        final_list = []
        for filename in os.listdir(in_folder):
            file = in_folder + '\\\\' + filename
            with open(file) as infile:
                final_list += list_of_words(preprocess(file))
            print(count)
            count += 1
        print('final_list length')
        print(len(final_list))
        word_count_dict = {}
        count = 0
        for key in final_list:
            word_count_dict[key] = final_list.count(key)
            print(count)
            count += 1
        outfile.write(repr(dict(sorted(word_count_dict.items(), key = itemgetter(1), reverse = True)[:100])))

#Word Freq Count test

def count_word(in_folder, output_file):
    entire_sent_list = []
    with open(output_file, 'a') as outfile:
        for filename in os.listdir(in_folder):
            file = in_folder + '\\\\' + filename
            with open(file) as infile:
                print(file)
                prep_file = preprocess(file)
                entire_sent_list += prep_file
                #print(entire_sent_list)
        outfile.write(repr(Counter(list_of_words(entire_sent_list))))
        return Counter(list_of_words(entire_sent_list))

#EDA_generator
#takes file path to folder with txt files- writes EDA findings in out_file path
#EDA findings: prints avg sentence length in terminal, top 100 words, word count dict (for word cloud)


# Tests:
# Two file test folder: C:\\Users\\kiana\\COSC490\\dir_test_folder\\TXTfolder1
## Getting average sent length
#print(average_sentence_length('C:\\Users\\kiana\\COSC490\\dir_test_folder\\TXTfolder1'))
## Getting word count dictionary
#print(word_count_dictionary('C:\\Users\\kiana\\COSC490\\dir_test_folder\\TXTfolder1', 'C:\\Users\\kiana\\COSC490\\dir_test_folder\\OUTfolder\\wcd_test1.txt'))
## Getting top 100 words
#print(highest_dict_value('C:\\Users\\kiana\\COSC490\\dir_test_folder\\TXTfolder1', 'C:\\Users\\kiana\\COSC490\\dir_test_folder\\OUTfolder\\hdv_test1.txt'))
## Counter function test
#print(count_word('C:\\Users\\kiana\\COSC490\\dir_test_folder\\TXTfolder1', 'C:\\Users\\kiana\\COSC490\\dir_test_folder\\OUTfolder\\counter_test.txt'))

#Final calls

#Article txt path: C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_txt_files
#Textbook txt path: C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_txt_files

## Getting average sent length
#print('avg sent length articles: ')
#print(average_sentence_length('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_txt_files'))
#print('avg sent length textbooks: ')
#print(average_sentence_length('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_txt_files'))

## Getting word count dictionary
#(word_count_dictionary('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_txt_files', 'C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_findings\\article_wcd.txt'))
#(word_count_dictionary('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_txt_files', 'C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_findings\\textbook_wcd.txt'))


## Getting top 100 words
#(highest_dict_value('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_txt_files', 'C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_findings\\article_hdv_test.txt'))
#(highest_dict_value('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_txt_files', 'C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_findings\\textbook_hdv_test.txt'))

## Counter function (serves both word count dict; can repurpose as top 100 words)
#(count_word('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_txt_files', 'C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\article_findings\\article_wcd.txt'))
#(count_word('C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_txt_files', 'C:\\Users\\kiana\\COSC490\\AI_Nephrologist\\textbook_findings\\textbook_wcd.txt'))

##To do:
#need to change count word to give top 100 words
#lemmatizer not working for all words? ask arvind