from encodings import utf_8
import numpy as np
import requests
#import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
import re
import math
#from sympy import li
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from operator import itemgetter

#to run
#pipenv run python preprocessor_list_of_sentences.py



#renal_biopsy.txt file path: C:\python310\pdf2txt\txt\pdfminer_renal_biopsy.txt 

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


#contractions removal function
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys())) #honeslty no clue what this is
def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_re.sub(replace, s)

#stopwords
stop_words = set(stopwords.words('english'))

#stemmer
def stem_words(text):
    return " ".join([PorterStemmer().stem(word) for word in text.split()])


#Preprocessing
###
# 1. Lower everything
# 2. Change contractions to two words: “don’t” -> “do not”
# 3. Removes all characters except A-Z, a-z, -, \n
# 4. Tokenizes by sentence (creates a list of sentences)
# 5. #need to replace '-\n' with ''
   #need to remove all other \n
    #need to remove stopwords

#removing the -\n from sentences in list
def hypen_problem_one(in_list):
    new_list = []
    for sentence in in_list:
        sentence = re.sub('-\n', '', sentence)
        new_list.append(sentence)
    return new_list

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


def average_sentence_length(in_list):
    sentence_lengths = []
    for sentence in in_list:
        tokens = word_tokenize(sentence)
        count = len(tokens)
        sentence_lengths.append(count)
    sentence_total = 0
    for num in sentence_lengths:
        sentence_total += num
    sentence_avg = sentence_total / len(sentence_lengths)
    return sentence_avg


def preprocess (in_file): #in_file is the unprocessed .txt file file path (str), outfile is where you want the preprocessed string to be written
    # open the data file
    file = open(in_file, encoding = 'utf8') #idk what encoding is but we needed this
    # read the file 
    data = file.read() #data is a string
    # close the file
    file.close()

    # lower text
    data = data.lower()

    #change contractions
    data = expand_contractions(data)
    
    

    #removing all charcters except for A-Z a-z .  - and space
    #comeback to hypens issues
    #looks just for -\n
    data = re.sub("[^A-Za-z .r\n-]", "", data)
    data = re.sub(' +', ' ', data)

    #tokenizes by sentence
    data = sent_tokenize(data)
    
    #gives me list of sentences, with \n charcters and =
    #looping through list now

    #fix hypen problem 1 (see above)
    data = hypen_problem_one(data)
    #fix hyoen problem two (see above)
    data = hypen_problem_two(data)
    #remove bad length words in sentences
    data = remove_strange_length_words(data)

    #convert to list of words
    word_data = []
    for sentence in data:
        tokens = word_tokenize(sentence)
        for item in tokens:
            word_data.append(item)


    return word_data

    
#test call to preprocess (before)
#preprocess('C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\preprocessed_neurotic_hypertension1.txt')


#have preprocess- need to call for each file in file_names and append output to the outfile
def append_preprocessed_files(file_names, output_file):
    with open(output_file, 'a') as outfile:
        final_list = []
        for names in file_names:
            with open(names) as infile:
                final_list += preprocess(names)
        outfile.write(repr(final_list))
#call to append every convereted pdf I have (4) and save to test_word_list.txt
#append_preprocessed_files(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renovascular_hypertension.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_neurotic_hypertension.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_antiglomerular_basement.txt'], 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\test_sent_list.txt')

#save to practice files
#append_preprocessed_files(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renovascular_hypertension.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_neurotic_hypertension.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_antiglomerular_basement.txt'], 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\better_word_list.txt')
#append_preprocessed_files(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt'], 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\test_sent_list_renal_biopsy.txt')



def append_preprocessed_files_list(file_names):
    list = []
    for names in file_names:
        with open(names) as infile:
            list = list + preprocess(names)
    return list
            
#average sentence length (removed stop words)
#print(average_sentence_length(append_preprocessed_files_list(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renovascular_hypertension.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_neurotic_hypertension.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_antiglomerular_basement.txt'])))

#outputs 11.674219228413962

def word_count_dictionary(file_names, output_file):
    with open(output_file, 'a') as outfile:
        final_list = []
        for names in file_names:
            with open(names) as infile:
                final_list += preprocess(names)
        word_count_dict = {}
        for key in final_list:
            word_count_dict[key] = final_list.count(key)
        outfile.write(repr(word_count_dict))
    return word_count_dict

def highest_dict_value(file_names, output_file):
    with open(output_file, 'a') as outfile:
        final_list = []
        for names in file_names:
            with open(names) as infile:
                final_list += preprocess(names)
        word_count_dict = {}
        for key in final_list:
            word_count_dict[key] = final_list.count(key)
        outfile.write(repr(dict(sorted(word_count_dict.items(), key = itemgetter(1), reverse = True)[:50])))
    print(dict(sorted(word_count_dict.items(), key = itemgetter(1), reverse = True)[:50]))


#word_count_dictionary(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renovascular_hypertension.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_neurotic_hypertension.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_antiglomerular_basement.txt'], 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\better_word_count_dictionary.txt')
#highest_dict_value(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renovascular_hypertension.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_neurotic_hypertension.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_antiglomerular_basement.txt'], 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\50_highest_word_counts.txt')
append_preprocessed_files(['C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renal_biopsy.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_renovascular_hypertension.txt','C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_neurotic_hypertension.txt', 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\pdfminer_antiglomerular_basement.txt'], 'C:\\Users\\kiana\\COSC490\\pdfminer_text_files\\better_word_list.txt')
