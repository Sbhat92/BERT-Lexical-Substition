#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 
import string

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    
    set_of_lemmas = set()
    synsets = wn.synsets(lemma, pos=pos)
    for i in range(len(synsets)):
        synset = synsets[i]
        lemmas = synset.lemmas()
        for lemma_ in lemmas:
            name = str(lemma_.name()) 
            if name != str(lemma):
                name = name.replace('_',"")
                set_of_lemmas.add(name)
    
    return list(set_of_lemmas)

def get_candidates_frequency(lemma, pos) -> List[str]:
    # Part 1
    
    set_of_lemmas = []
    synsets = wn.synsets(lemma, pos=pos)
    for i in range(len(synsets)):
        synset = synsets[i]
        lemmas = synset.lemmas()
        for lemma_ in lemmas:
            name = str(lemma_.name()) 
            if name != str(lemma):
                name = name.replace('_',"")
                for i in set_of_lemmas:
                    if i[0] == name:
                        i[1]+= 1
                        continue
                
                set_of_lemmas.append([name,1])
    
    return list(set_of_lemmas)


def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    frequency = {}
    lemma = context.lemma
    pos = context.pos    
    lemmas = wn.lemmas(lemma, pos)     
    for i in lemmas:        
        s1 = i.synset()        
        lemmas = s1.lemmas()
        for lemma_ in lemmas:
            #print(lemma_,lemma_.name(),lemma_.count())
            name = str(lemma_.name())
            name = name.replace('_'," ")
            if name != str(context.lemma):                
                if name in frequency:
                    frequency[name] += lemma_.count()
                    #print("first",lemma_,lemma_.count())
                else:
                    frequency[name] = lemma_.count()
                    #print("second",lemma_,lemma_.count())
    ans = ''    
    freq = 0
    
    for i in frequency:
        if frequency[i] >= freq:
            ans = i
            freq = frequency[i]
        
    
    return ans
    
    return None # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:
     
    from nltk import word_tokenize
    
           
    ##########################
    
    lemma = context.lemma
    pos = context.pos
    list_ = wn.lemmas(lemma, pos)    
    lemma_og = lemma
      
    ans = 'smurf'
    #print(context)
    
    stop_words = stopwords.words('english')
    con_ = context.left_context + context.right_context
    con = [word.lower() for word in con_ 
             if word not in stop_words and word.isalpha()]
    #print("context",con)
    
    #print(str(context.word_form))
    i=0
    while i < len(con):
        if con[i] == str(context.word_form) or con[i] in stop_words:
            con.remove(con[i])
        else:
            i+=1
            
    #print(con)
    score_max = 0
    for l in list_:
        s = l.synset()
        lemmas = s.lemmas()
        #print(lemmas)
        for lemma_ in lemmas:
            if lemma_.name() != lemma_og:
                score1=0
                score2=0
                score3=0
                test = []
                #find term 1--overlap between context and all of u
                ###
                defi = word_tokenize(s.definition())
                test += defi       
                examples = s.examples()
                for example in examples:
                    test+=word_tokenize(example)
                hyper = s.hypernyms()

                for syn in hyper:            
                    defi = word_tokenize(syn.definition())
                    test+=defi
                    examples = syn.examples()
                    for example in examples:
                        test+=word_tokenize(example)
                #print("test",test)
                ###
                for i in con:
                    if i in test:
                        score1 += 1


                score3 = lemma_.count()

                lemmas_i = s.lemmas()
                #print("inside",lemmas_i)
                for lemma_i in lemmas_i: 
                    #print(lemma_i.name(),lemma_og)
                    name = lemma_i.name()
                    name.replace("_"," ")
                    if name == lemma_og:
                        score2 += 1
                        #print("added")
                #print(score1,score2,score3,lemma_i,lemma_og)
                score = 1000*score1 +100*score2+ score3
                if score > score_max:
                    ans = str(lemma_.name())
                    score_max = score
    
    return ans
    
    
    return None #replace for part 3        


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        possibilities = get_candidates(context.lemma, context.pos)
        #print("possibilities",possibilities)
        sim = 0
        for i in possibilities:
            lemma = str(context.lemma)
            lemma.replace("_"," ")
            
            try:
                new = self.model.similarity(i,lemma)
            except:
                continue
            if new >= sim:
                sim = new
                ans = i
        return ans
        return "smurf"
        
        
        return None # replace for part 4


class BertPredictor(object):
    
    def __init__(self, filename): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:                
        
        con = " ".join(context.left_context) +  " [MASK] " + " ".join(context.right_context)
        #print(con)
        #tokened = self.tokenizer.tokenize(con)
        
        input_toks = self.tokenizer.encode(con)
        #print(tokened)
        tokened = self.tokenizer.convert_ids_to_tokens(input_toks)
        #print(tokened)
        for i in range(len(tokened)):
            if tokened[i] == "[MASK]":
                ind = i
                break
        possibilities = get_candidates(context.lemma, context.pos)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][ind])[::-1]
        list1 = self.tokenizer.convert_ids_to_tokens(best_words)
        for i in list1:
            #print(i,ind,tokened)
            if i in possibilities:
                return i
        
        return "Smurf" # replace for part 5


class NewBertPredictor(object):
    #Not used but this is my idea: if we can somehow test the hyperparameters,
    #maybe this model would work well. Right now I am just using a 1:1 weight.
    #we can learn appropriate weights to make it work better
    def __init__(self, filename): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.modelw = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True) 

    def predict(self, context : Context) -> str:                
        import math
        con = " ".join(context.left_context) +  " [MASK] " + " ".join(context.right_context)        
        input_toks = self.tokenizer.encode(con)
        
        tokened = self.tokenizer.convert_ids_to_tokens(input_toks)        
        for i in range(len(tokened)):
            if tokened[i] == "[MASK]":
                ind = i
                break
        possibilities = get_candidates(context.lemma, context.pos)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][ind])[::-1]
        list1 = self.tokenizer.convert_ids_to_tokens(best_words)
        ansd = {}
        for i in range(len(list1)):            
            if list1[i] in possibilities:
                try:
                    sim = self.modelw.similarity(list1[i],context.lemma)
                    ansd[list1[i]] = -i/len(list1) + sim/len(list1)
                except:
                    ans[list1[i]] = -i/len(list1)
        maxi = -math.inf
        ans = 'smurf'
        for i in ansd:
            if ansd[i] > maxi:
                ans = i
                maxi = ansd[i]
        
        return ans
        return "Smurf" # replace for part 5


class part6(object):
    ##Similar problem for this method.. not sure how to weight them correctly
    def __init__(self, filename):        
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        
    def predict(self, context : Context) -> str:
        from nltk import word_tokenize


        ##########################

        lemma = context.lemma
        pos = context.pos
        list_ = wn.lemmas(lemma, pos)    
        lemma_og = lemma

        ans = 'smurf'
        #print(context)

        stop_words = stopwords.words('english')
        con_ = context.left_context + context.right_context
        con = [word.lower() for word in con_ 
                 if word not in stop_words and word.isalpha()]
        #print("context",con)

        #print(str(context.word_form))
        i=0
        while i < len(con):
            if con[i] == str(context.word_form) or con[i] in stop_words:
                con.remove(con[i])
            else:
                i+=1

        #print(con)
        score_max = 0
        for l in list_:
            s = l.synset()
            lemmas = s.lemmas()
            #print(lemmas)
            for lemma_ in lemmas:
                if lemma_.name() != lemma_og:
                    score1=0
                    score2=0
                    score3=0
                    test = []
                    #find term 1--overlap between context and all of u
                    ###
                    defi = word_tokenize(s.definition())
                    test += defi       
                    examples = s.examples()
                    for example in examples:
                        test+=word_tokenize(example)
                    hyper = s.hypernyms()

                    for syn in hyper:            
                        defi = word_tokenize(syn.definition())
                        test+=defi
                        examples = syn.examples()
                        for example in examples:
                            test+=word_tokenize(example)
                    #print("test",test)
                    ###
                    for i in con:
                        if i in test:
                            score1 += 1


                    score3 = lemma_.count()

                    lemmas_i = s.lemmas()
                    #print("inside",lemmas_i)
                    for lemma_i in lemmas_i: 
                        #print(lemma_i.name(),lemma_og)
                        name = lemma_i.name()
                        name.replace("_"," ")
                        if name == lemma_og:
                            score2 += 1
                            #print("added")
                    #print(score1,score2,score3,lemma_i,lemma_og)
                    score4 = 0
                    try:
                        score4 = self.model.similarity(lemma_.name(),context.lemma)
                    except:
                        pass
                    score = 1000*score1 + 1000*score4 + 100*score2+ score3
                    if score > score_max:
                        ans = str(lemma_.name())
                        score_max = score

        return ans


        return None #replace for part 3 


def part7(context):    
    #keep a threshold of 1000 for scoremax, if we hit it, return the result of part3, if not, 
    #part 3 does not do a good job, lets try part 2
    #results in an imporvement of up to .150 for different hyperparameters..
    #still does not beat bert, but beats both part2 and part3
    from nltk import word_tokenize

    lemma = context.lemma
    pos = context.pos
    list_ = wn.lemmas(lemma, pos)    
    lemma_og = lemma

    ans = 'smurf'
    #print(context)

    stop_words = stopwords.words('english')
    con_ = context.left_context + context.right_context
    con = [word.lower() for word in con_ 
             if word not in stop_words and word.isalpha()]
    #print("context",con)

    #print(str(context.word_form))
    i=0
    while i < len(con):
        if con[i] == str(context.word_form) or con[i] in stop_words:
            con.remove(con[i])
        else:
            i+=1

    #print(con)
    score_max = 0
    for l in list_:
        s = l.synset()
        lemmas = s.lemmas()
        #print(lemmas)
        for lemma_ in lemmas:
            if lemma_.name() != lemma_og:
                score1=0
                score2=0
                score3=0
                test = []
                #find term 1--overlap between context and all of u
                ###
                defi = word_tokenize(s.definition())
                test += defi       
                examples = s.examples()
                for example in examples:
                    test+=word_tokenize(example)
                hyper = s.hypernyms()

                for syn in hyper:            
                    defi = word_tokenize(syn.definition())
                    test+=defi
                    examples = syn.examples()
                    for example in examples:
                        test+=word_tokenize(example)
                #print("test",test)
                ###
                for i in con:
                    if i in test:
                        score1 += 1


                score3 = lemma_.count()

                lemmas_i = s.lemmas()
                #print("inside",lemmas_i)
                for lemma_i in lemmas_i: 
                    #print(lemma_i.name(),lemma_og)
                    name = lemma_i.name()
                    name.replace("_"," ")
                    if name == lemma_og:
                        score2 += 1
                        #print("added")
                #print(score1,score2,score3,lemma_i,lemma_og)
                score = 1000*score1 +100*score2+ score3
                
                if score > score_max:
                    ans = str(lemma_.name())
                    score_max = score
    #print(score_max)
    if score_max < 1000:
        predictor = wn_frequency_predictor(context)
        return predictor

    return ans


    return None #replace for part 3      


# +
class part8(object):
    #for each posibility, if we beat the score, we check our new score calculated using method 3
    #and try to weight each possibility using that, this is one approach I thought of that did
    #not yeild good results
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        possibilities = get_candidates(context.lemma, context.pos)
        #print("possibilities",possibilities)
        sim = 0
        lemma_og = context.lemma
        stop_words = stopwords.words('english')
        con_ = context.left_context + context.right_context
        con = [word.lower() for word in con_ 
                 if word not in stop_words and word.isalpha()]
        #print("context",con)

        #print(str(context.word_form))
        i=0
        while i < len(con):
            if con[i] == str(context.word_form) or con[i] in stop_words:
                con.remove(con[i])
            else:
                i+=1
        
        score_max = 0
        for i in possibilities:
            lemma = str(context.lemma)
            lemma.replace("_"," ")
            
            try:
                new = self.model.similarity(i,lemma)
            except:
                continue
                
            if new >= sim:
                sim = new
                list_ = wn.lemmas(i) 
                for l in list_:
                    s = l.synset()
                    lemmas = s.lemmas()
                    #print(lemmas)
                    for lemma_ in lemmas:
                        if lemma_.name() != lemma_og:
                            score1=0
                            score2=0
                            score3=0
                            test = []
                            #find term 1--overlap between context and all of u
                            ###
                            defi = tokenize(s.definition())
                            test += defi       
                            examples = s.examples()
                            for example in examples:
                                test+=tokenize(example)
                            hyper = s.hypernyms()

                            for syn in hyper:            
                                defi = tokenize(syn.definition())
                                test+=defi
                                examples = syn.examples()
                                for example in examples:
                                    test+=tokenize(example)
                            #print("test",test)
                            ###
                            for i in con:
                                if i in test:
                                    score1 += 1


                            score3 = lemma_.count()

                            lemmas_i = s.lemmas()
                            #print("inside",lemmas_i)
                            for lemma_i in lemmas_i: 
                                #print(lemma_i.name(),lemma_og)
                                name = lemma_i.name()
                                name.replace("_"," ")
                                if name == lemma_og:
                                    score2 += 1
                                    #print("added")
                            #print(score1,score2,score3,lemma_i,lemma_og)
                            score = 1000*score1 +100*score2+ score3
                            if score > score_max:
                                ans = str(lemma_.name())
                                score_max = score

        return ans
        return "smurf"
        
        
        
# -

class part9(object):
    #tried getting frequency of word as well in the get_candidates and tried including that 
    #frequency in the weighting found in model.similarity.. once again the issue of weighting comes up
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        possibilities = get_candidates_frequency(context.lemma, context.pos)
        #print("possibilities",possibilities)
        sim = 0
        summ = 0
        ans = "smurf"
        for i in possibilities:
            summ+=i[1]
            
        for i in possibilities:
            lemma = str(context.lemma)
            lemma.replace("_"," ")
            
            try:
                new = self.model.similarity(i[0],lemma) + i[1]/summ
            except:
                continue
            if new >= sim:
                sim = new
                ans = i[0]
        
        return ans
        
        
        
        return None # replace for part 4

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        predictor = BertPredictor(W2VMODEL_FILENAME)
        #prediction = part7(context)
        prediction = predictor.predict(context) 
        #prediction = part7(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))



