# coding=utf-8
# -*- coding: UTF-8 -*- 
"""
Author  : Erutan
Email   : erutan@pku.edu.cn, erutan.pkuicst@gmail.com
GitHub  : https://github.com/Erutan-pku

version : V1.0.0
date    : April 22nd, 2017

Python 2.7.10
"""
import csv
import codecs
import re
import pandas as pd
import numpy as np
import sys
import json

#word2vec
class W2V :
    w2i = {}
    i2v = {}
    wset = set()
    size = 0
    def __init__(self, wordVecFile, ignoreEndingLength=1, split=' ') :
        input = codecs.open(wordVecFile, encoding='utf-8')

        start =True
        for line in input:
            if start :
                start = False
                continue

            word = line.split(split)[0]
            vec = line2list(line[:-ignoreEndingLength], split=split, convert=float, start=1)

            self.w2i[word] = len(self.w2i)
            self.i2v[self.w2i[word]] = vec
        self.wset = set(self.w2i.keys())
        self.size = len(self.wset)

    def getWordID(self, word) :
        return self.w2i[word] if word in self.wset else self.w2i['__NWord__']
    def getSequence(self, wordList, max_len) :
        wordt = ['__<\s>__', '__<\s>__']
        for word in wordList :
            wordt.append(word)
        wordt += ['__<\s>__', '__<\s>__']
        for word in wordList[::-1] :
            wordt.append(word)
        wordt += ['__<\s>__', '__<\s>__']
        while len(wordt) < max_len :
            wordt.append('__<0>__')

        ret = []
        for i in range(max_len) :
            ret.append(self.getWordID(wordt[i]))
        return ret
    def getCosine(self, w1, w2) :
        if not w1 in self.wset and not w2 in self.wset :
            return 0.0
        v1 = np.array(self.i2v[self.getWordID(w1)])
        v2 = np.array(self.i2v[self.getWordID(w2)])

        dot_num = float(np.dot(v1,v2))
        de_nom = np.linalg.norm(v1) * np.linalg.norm(v2)  
        cos = dot_num / de_nom 
        sim = 0.5 + 0.5 * cos
        return sim
    def getMaxCos(self, w1, wset) :
        max_vt = 0
        wt = ''
        for w_i in wset :
            vt = self.getCosine(w1, w_i)
            if vt > max_vt :
                max_vt = vt
                wt = w_i
        return max_vt, wt
    def getMatrix(self) :
        ret = []
        for i in range(self.size) :
            ret.append(self.i2v[i])
        return np.array(ret)

#counter
class Counter :
    def __init__(self, dict_in=None, listStart=1) :
        if dict_in is None :
            self.count_hash = {}
        else :
            assert type(dict_in) in [list, dict]
            if type(dict_in) is dict :
                self.count_hash = dict_in
            elif type(dict_in) is list :
                self.count_hash = {}
                for i, num in enumerate(dict_in) :
                    self.count_hash[i + listStart] = num
    def __len__(self):
        return len(self.count_hash)
    def get(self, name) :
        if not name in self.count_hash :
            return 0
        else :
            return self.count_hash[name]
    def count(self, name, value = 1) :
        if not name in self.count_hash :
            self.count_hash[name] = 0
        self.count_hash[name] += value
    def combine(self, another) :
        assert isinstance(another, Counter)
        for key in another.count_hash :
            if not key in self.count_hash :
                self.count_hash[key] = 0
            self.count_hash[name] += another[key]
    def getSortedList(self, least=True) :
        ret_list = [(key, self.count_hash[key]) for key in self.count_hash if least is True or self.count_hash[key] >= least]
        ret_list = sorted(ret_list, key=lambda x:x[1], reverse=True)
        return ret_list

# deal
def list2line(inputList, split=u' ', convert=unicode) :
    ret = u''
    if len(inputList) == 0 :
        return ret

    ret = convert(inputList[0])
    for i in range(1, len(inputList)) :
        ret += split
        ret += convert(inputList[i])
    return ret
def line2list(inputString, split=u' ', convert=None, start=0, end=None) :
    ret = []
    ls = inputString.split(split)
    ls = ls[start:] if end == None else ls[start:end]
    
    for ls_i in ls :
        if convert == None :
            ret.append(ls_i)
        else :
            ret.append(convert(ls_i))
    return ret
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring
def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring
def list2dict(listName, mainKey) :
    # list of json 2 dict
    retDict = {}
    for list_i in listName :
        retDict[list_i[mainKey]] = list_i
    return retDict
def dictValueList(dictName) :
    return [dictName[key] for key in dictName.keys()]
def dict2list(dictName) :
    return [[key, dictName[key]] for key in dictName.keys()]
# input
def loadLists(filename, convert=None, retTypeSet=False, ignoreEndingLength=1, ignoreFirstLine=False) :
    input = codecs.open(filename, encoding='utf-8')
    retList = []
    
    start=True
    for line in input : 
        if start and ignoreFirstLine :
            start = False
            continue

        lt = line[:-ignoreEndingLength]
        if convert == None :
            retList.append(lt)
        else :
            retList.append(convert(lt))

    if retTypeSet :
        retList = set(retList)
    return retList
def loadListofDict(filename, output_type='None', ignoreEndingLength=1) :
    assert output_type in ['None', 'list', 'dict']

    input = codecs.open(filename, encoding='utf-8')
    mainKey = None
    ret_list = None

    start = True
    stringBuffer = ''
    for line in input :
        line = line[:-ignoreEndingLength]

        if start == True :
            start = False
            mainKey = line.split('\t')[1]
            if output_type == 'None' :
                if mainKey == 'None' :
                    output_type = 'list'
                else :
                    output_type = 'dict'
            if output_type == 'dict' :
                assert not mainKey == 'None'
            ret_list = [] if output_type == 'list' else {}
            
            continue

        stringBuffer += line.strip()
        if line == '}' :
            json_i = json.loads(stringBuffer)
            if output_type == 'list' :
                ret_list.append(json_i)
            else :
                assert mainKey in json_i.keys()
                ret_list[json_i[mainKey]] = json_i
            stringBuffer = ''
            

    return ret_list

# output
def writeList(filename, listName, convert=str) :
    output = codecs.open(filename, "w", "utf-8")
    for li in listName :
        output.write(convert(li) + '\n')
    output.flush()
    output.close()
def writeListofDict(filename, listName, mainKey='None', readable=True) :
    output = codecs.open(filename, "w", "utf-8")
    output.write('mainKey:\t%s\n'%(mainKey))
    for t_dict in listName :
        assert ((mainKey == 'None') or (mainKey in t_dict.keys()))
        output.write(json.dumps(t_dict, ensure_ascii=False, indent=4 if readable else None) + '\n')
    output.flush()
    output.close()

# special
def loadQIDs(filename, split='\t', ignoreFirstLine=False) :
    input = codecs.open(filename, encoding='utf-8')
    retSet = set()
    
    start = True
    for line in input : 
        if start and ignoreFirstLine :
            start = False
            continue

        qid = int(line.split(split)[0])
        retSet.add(qid)
    return retSet
def getKeySet(listName, keyName, convert=None) :
    ret = set()
    for dict_t in listName :
        if convert == None :
            ret.add(dict_t[keyName])
        else :
            ret.add(convert(dict_t[keyName]))
    return ret


""" marks :
http://stackoverflow.com/questions/24005761/eof-inside-string-in-big-data
    solution : deal with unmatched " ect. in that line.
"""

if __name__ == '__main__' :
    pass