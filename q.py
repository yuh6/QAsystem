import spacy
import en_core_web_lg

import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import json
from sklearn.feature_extraction.text import CountVectorizer  # build matrix
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict, Counter
import numpy as np
from collections import defaultdict, Counter
from math import log
import string
import re
import csv
from nltk.tokenize import word_tokenize


with open('documents.json') as data_file:
    data = json.load(data_file)

stopwords = set(nltk.corpus.stopwords.words('english'))
#nlp = spacy.load('en')
#nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load("en-core-web-lg")
#nlp = spacy.load('en-core-web-lg')
nlp = en_core_web_lg.load()

def remove_non_ascii_2(text):
    return re.sub(r'[^\x00-\x7F]', ' ', text)

def tokenize(sentence):
    word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
    tokenized_sentence = word_tokenizer.tokenize(sentence)
    for word in tokenized_sentence:
        if word in string.punctuation:
            tokenized_sentence.remove(word)
        elif word in stopwords:
            tokenized_sentence.remove(word)
    return tokenized_sentence

def preprocess_query(query):
    query = query.lower()
    word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
    tokenized_query = word_tokenizer.tokenize(query)
    tokenized_query = lemmitize(tokenized_query)
    tokenized_query.remove('?')
    return tokenized_query

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None  # for easy


def lemmitize(list):
    word_list = []
    for word, tag in nltk.pos_tag(list):
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # not supply tag in case of None
            lemma = lemmatizer.lemmatize(word)
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
        word_list.append(lemma)
    return word_list

def score_BM25n(N, ft, b, k1, fdt, docl, avdocl, k3, fqt):
    # first = log(((r + 0.5)/(R - r + 0.5))/((ft - r + 0.5)/(N - ft - R + r + 0.5)))
    first = log(N / ft)
    K = k1 * ((1 - b) + b * (float(docl) / float(avdocl)))
    second = ((k1 + 1) * fdt) / (K + fdt)
    third = ((k3 + 1) * fqt) / (k3 + fqt)
    return first * second * third


def get_VSM_inverted_index(docid):
    inital_sentences = []
    all_sentences = []
    length_list = []
    document_text = data[docid]['text']
    for paragraph in document_text:
        paragraph_sent = sent_tokenize(paragraph)
        for sentence in paragraph_sent:
            inital_sentences.append(sentence)
            sentence = sentence.lower()
            sentence = tokenize(sentence)
            length_list.append(len(sentence))  # find the length of each sentence
            sentence = lemmitize(sentence)
            new_sentence = ' '.join(sentence)
            all_sentences.append(new_sentence)  # buid all the sentence
            while '' in all_sentences:
                all_sentences.remove('')
                length_list.remove(0)

    vectorizer = CountVectorizer()
    VSM = vectorizer.fit_transform(all_sentences).toarray()
    term_dic = {v: k for k, v in vectorizer.vocabulary_.items()}
    list_of_index = np.transpose(np.nonzero(VSM))
    vsm_inverted_index = defaultdict(lambda: [])
    for x, y in list_of_index:
        term = term_dic[y]
        count = VSM[x, y]
        vsm_inverted_index[term].append((x, count))# build the inverted_index
    return length_list, vsm_inverted_index, all_sentences, inital_sentences


def run_query(query, docid):
    length_list, vsm_inverted_index, all_sentences, inital_sentences = get_VSM_inverted_index(docid)
    query_result = dict()
    query = preprocess_query(query)  # the stop words have been removed
    for term in query:
        if term in vsm_inverted_index:
            for sentid, freq in vsm_inverted_index[term]:
                score = score_BM25n(N=len(length_list), ft=len(vsm_inverted_index[term]), b=0.75, k1=1.2,
                                    fdt=freq, docl=length_list[sentid], avdocl=np.mean(length_list), k3=2,
                                    fqt=1)  # calculate score
                if sentid in query_result:  # this document has already been scored once
                    query_result[sentid] += score
                else:
                    query_result[sentid] = score
    sorted_list = sorted(query_result.items(), key=lambda x: x[1], reverse=True)
    target_sentenceid = sorted_list[0][0]
    return target_sentenceid


def preprocess_query(query):
    query = query.lower()
    word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
    tokenized_query = word_tokenizer.tokenize(query)
    tokenized_query = lemmitize(tokenized_query)
    if '?' in tokenized_query:
        tokenized_query.remove('?')
    return tokenized_query


def what_question(tokenized_query): # 'what' answer type classification
    questionwords = ['date', 'year', 'person', 'scientist', 'percentage', 'large', 'small', 'little', 'company',
                     'organization', 'institution', 'city', 'country', 'state', 'place', 'location', 'nation',
                     'frequency', 'rank', 'county', 'price', 'level']
    NER = ['DATE', 'DATE', 'PERSON', 'PERSON', 'PERCENT', 'PERCENT', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'ORG', 'GPE',
           'GPE', 'GPE', 'LO', 'LO', 'GPE', 'CARDINAL', 'ORDINAL', 'GPE', 'MONEY', 'ORDINAL']
    index = tokenized_query.index('what')
    index_list = []
    for word in questionwords:
        if word in tokenized_query:
            query_wordidex = tokenized_query.index(word)  # find the questionword postition
            index_list.append(query_wordidex)
        else:
            index_list.append(100)  # put the position in list
    index_cloest = min(index_list)
    if index_cloest != 100:  # the samllest position
        afterword = questionwords[index_list.index(index_cloest)]
        # find the word
        index_NER = questionwords.index(afterword)  # find the position in the NER
        type = NER[index_NER]
    else:
        type = 'NONE'
    return type


def which_question(tokenized_query):  # 'which' answer type classification
    questionwords = ['date', 'year', 'person', 'scientist', 'percentage', 'large', 'small', 'little', 'company',
                     'organization', 'institution', 'city', 'country', 'state', 'place', 'location', 'nation',
                     'frequency', 'rank', 'county', 'price', 'level']

    NER = ['DATE', 'DATE', 'PERSON', 'PERSON', 'PERCENT', 'PERCENT', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'ORG', 'GPE',
           'GPE', 'GPE', 'LO', 'LO', 'GPE', 'CARDINAL', 'ORDINAL', 'GPE', 'MONEY', 'ORDINAL']

    index = tokenized_query.index('which')
    index_list = []
    for word in questionwords:
        if word in tokenized_query:
            query_wordidex = tokenized_query.index(word)  # find the questionword postition
            index_list.append(query_wordidex)
        else:
            index_list.append(100)  # put the position in list

    index_cloest = min(index_list)
    if index_cloest != 100:  # the samllest position
        afterword = questionwords[index_list.index(index_cloest)]# find the word
        index_NER = questionwords.index(afterword)  # find the position in the NER

        type = NER[index_NER]

    else:
        type = 'NONE'
    return type


def question_type(query): # All question types classification
    tokenized_query = preprocess_query(query)
    askwords = ['what', 'who', 'where', 'which', 'how', 'whose', 'whom', 'when']
    index_list = []
    for word in askwords:
        if word in tokenized_query:
            index = tokenized_query.index(word)
            index_list.append(index)
        else:
            index_list.append(100)
    if min(index_list) == 100:
        type = 'NONE'
    else:
        askwords_position = index_list.index(min(index_list))
        askword = askwords[askwords_position]

        if askword == 'who' or askword == 'whose' or askword == 'whom':
            type = "PERSON"
        elif askword == 'what':
            type = what_question(tokenized_query)

        elif askword == 'which':
            type = which_question(tokenized_query)

        elif askword == 'when':
            type = 'DATE'

        elif askword == 'how':
            index = tokenized_query.index('how')
            if tokenized_query[index + 1] == 'many':
                type = "QUANTITY"
            elif tokenized_query[index + 1] == 'long':
                type = "TIME"
            elif tokenized_query[index + 1] == 'much':
                type = 'MONEY'
            else:
                type = 'NONE'
        elif askword == 'where':
            type = 'GPE'
        #             if 'level'in tokenized_query :
        #                 type == 'ORDINAL'
        #             else:
        #                 type == 'GPE'
        else:
            type == 'NONE'
    # print type
    return type


def choose_token(query, answer_list, target_sent):  # Selecting: Choose patterns from the candidates
    string7 = ''

    #     print answer_list
    tokenized_query = tokenize(query)
    final_query = lemmitize(tokenized_query)
    tokenized_sentence = tokenize(target_sent)
    final_sent = lemmitize(tokenized_sentence)
    final_answer = answer_list

    answer_pos_dis = []
    for answer in final_answer:
        string = answer.split(' ')
        answer_start = string[0]
        answer_end = string[-1]

        dis = 0
        s = None
        e = None
        if answer_start in tokenized_sentence:
            s = tokenized_sentence.index(answer_start)

        if answer_end in tokenized_sentence:
            e = tokenized_sentence.index(answer_end)

        if s is not None and e is not None:
            for query_word in final_query:
                if query_word in final_sent:
                    query_word_pos = final_sent.index(query_word)
                    dis += min(abs(s - query_word_pos), abs(e - query_word_pos))
            answer_pos_dis.append(dis)
        else:
            answer_pos_dis.append(999999)

    answer_index = answer_pos_dis.index(min(answer_pos_dis))
    string1 = answer_list[answer_index]
    answer_pos_dis.remove(min(answer_pos_dis))
    final_answer.remove(final_answer[answer_index])

    if len(answer_pos_dis) != 0:
        answer_index = answer_pos_dis.index(min(answer_pos_dis))
        string2 = string1 + ' ' + answer_list[answer_index]
        answer_pos_dis.remove(min(answer_pos_dis))
        final_answer.remove(final_answer[answer_index])
        if len(answer_pos_dis) != 0:
            answer_index = answer_pos_dis.index(min(answer_pos_dis))
            string3 = string2 + ' ' + answer_list[answer_index]
            answer_pos_dis.remove(min(answer_pos_dis))
            final_answer.remove(final_answer[answer_index])
            if len(answer_pos_dis) != 0:
                answer_index = answer_pos_dis.index(min(answer_pos_dis))
                string4 = string3 + ' ' + answer_list[answer_index]
                answer_pos_dis.remove(min(answer_pos_dis))
                final_answer.remove(final_answer[answer_index])
                if len(answer_pos_dis) != 0:
                    answer_index = answer_pos_dis.index(min(answer_pos_dis))
                    string5 = string4 + ' ' + answer_list[answer_index]
                    answer_pos_dis.remove(min(answer_pos_dis))
                    final_answer.remove(final_answer[answer_index])
                    if len(answer_pos_dis) != 0:
                        answer_index = answer_pos_dis.index(min(answer_pos_dis))
                        string6 = string5 + ' ' + answer_list[answer_index]
                        answer_pos_dis.remove(min(answer_pos_dis))
                        final_answer.remove(final_answer[answer_index])
                        if len(answer_pos_dis) != 0:
                            answer_index = answer_pos_dis.index(min(answer_pos_dis))
                            string7 = string5 + ' ' + answer_list[answer_index]
                            answer_pos_dis.remove(min(answer_pos_dis))
                            final_answer.remove(final_answer[answer_index])
                        else:
                            string7 = string6.strip()
                    else:
                        string7 = string5.strip()
                else:
                    string7 = string4.strip()
            else:
                string7 = string3.strip()
        else:
            string7 = string2.strip()
    else:
        string7 = string1.strip()

    return string7.strip()


def answer_question(query, docid):   #Define the answering process
    enties = []
    chunks = []

    answer_text = []
    length_list, vsm_inverted_index, all_sentences, inital_sentences = get_VSM_inverted_index(docid)
    sentid = run_query(query, docid)
    target_sentence = inital_sentences[sentid]
    # print target_sentence
    target_sentence = remove_non_ascii_2(target_sentence)
    qustype = question_type(query)

    doc = nlp(inital_sentences[sentid])

    for ent in doc.ents:                        # Name entities matching
        enties.append((ent.text, ent.label_))

    for chunk in doc.noun_chunks:
        chunks.append(chunk.text)


    for text, ent in enties:
        new_text = lemmitize(tokenize(text))

        new_query = lemmitize(tokenize(query))
        if qustype == ent and len(set((new_text)) & set(new_query)) == 0 and text.lower() not in answer_text:
            #             token = remove_non_ascii_2(text)
            #             answer_text += token.lower()
            answer_text.append(text.lower())
    # print  (answer_text)
    if len(answer_text) == 0:        # Chunk matching
        if len(chunks) != 0:
            for i in range(0, len(chunks)):
                new_text = lemmitize(tokenize(chunks[i]))
                new_query = lemmitize(tokenize(query))
                chunk = tokenize(chunks[i].lower())

                remove_list = []
                for j in range(len(chunk[:])):
                    if new_text[j] in stopwords or new_text[j] in set((new_text)) & set(new_query):
                        remove_list.append(chunk[j])
                for t in remove_list:
                    chunk.remove(t)
                chunk = ' '.join(t for t in chunk).strip()

                if chunk not in answer_text:
                    answer_text.append(chunk)

    if len(answer_text) == 0:                #dependency parsing
        for token in doc:
            new_text = lemmitize(token.text)
            new_query = lemmitize(tokenize(query))

            if token.head.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
                if new_text not in new_query:
                    if token.text.lower() not in answer_text:
                        answer_text.append(token.text.lower())

    #                     answer_text += token.lower()
    if len(answer_text) == 0:
        result = ''
        for token in doc:
            new_text = lemmitize(token.text)
            #             print (new_text)
            #             print (token.text)
            # print (result)
            new_query = lemmitize(tokenize(query))
            # if token.text not in query:
            if new_text not in new_query and token.pos != 'ADP':
                if new_text not in stopwords and token.text not in result:
                    #                     print new_text
                    #                     print token.text
                    #                     print result
                    if token.text not in string.punctuation:
                        result += token.text + ' '
        result = result.strip()



    elif len(answer_text) == 1:
        result = answer_text[0]
    else:
        for possible_ans in answer_text:
            if possible_ans in stopwords and possible_ans not in answer_text:
                answer_text.remove(possible_ans)

        result = choose_token(query, answer_text, target_sentence)
    # print (answer_text)
    return result


# with open("answer.csv", "a") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['id', 'answer'])
#     with open('testing.json') as question_file:
#         question_data = json.load(question_file)
#
#         for tmp_data in question_data:
#             id = tmp_data['id']
#             query = tmp_data['question']
#             docid = tmp_data['docid']
#             answer = answer_question(query, docid)      # answer
#             print(query, answer)
#             writer.writerow([id, answer])

query = input()
answer = answer_question(query, 0)
print(answer)


