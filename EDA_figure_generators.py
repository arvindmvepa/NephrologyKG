import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud















##Wordcloud
#takes dictionary and generates wordcloud of the 300 most frequently used words
def create_wordcloud(wordcloud_dict):
    wordcloud = WordCloud(font_path= "arial", background_color="black", colormap= "tab20c", width=2000, height=2000, max_words=300, relative_scaling=0.5, normalize_plurals=False).generate_from_frequencies(wordcloud_dict)

    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.show()

#Create article wordcloud (tab20c)
#create_wordcloud(textbook_dict)

#Create textbook wordcloud (coolwarm)
#create_wordcloud(article_dict)


##Top n words
#takes dictionary and number n words and returns sorted dictionary of top n words
def top_n_words(in_dict, n):
    dict_list = []
    for key in sorted(in_dict.items(), key = lambda item: item[1], reverse = True)[:n]:
        dict_list += [key]
    top_n_dict = dict(dict_list)
    return top_n_dict

#Top 50 words in articles
#print('Top 50 words in atricles: ')
#print(top_n_words(article_dict, 50))

#Top 50 words in textbooks
#print('Top 50 words in textbooks: ')
#print(top_n_words(textbook_dict, 50))



##Histogram

def histogram_generator(dict):
    #plt.rc('xtick', labelsize=10) 
    #plt.rc('ytick', labelsize=10) 
    plt.figure(figsize=(20, 10))  # width:20, height:3
    plt.rc('font', size = 8)
    plt.rc('axes', titlesize = 6)
    plt.xticks(rotation = 45)
    plt.bar(range(len(dict)), dict.values(), align='edge', width=0.3) 
    plt.bar(dict.keys(), dict.values(), align = 'edge', width = 0.5,  color='b')
    plt.show()

#Create article histogram
histogram_generator(top_n_words(textbook_dict, 50))

#Create textbook histogram
#histogram_generator(top_n_words(textbook_dict, 50))