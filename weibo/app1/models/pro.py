'''
Created on 2018年5月25日

@author: zhao
'''
import pandas as pd
import json
import jieba
import sys 
from sklearn.feature_extraction.text import CountVectorizer #词袋模型
from sklearn import metrics #模型评估
from sklearn.naive_bayes import MultinomialNB
import os

class Model(object):
    """docstring for ClassName"""
    def __init__(self):

        super(Model, self).__init__()
       
        self.make_data()

    def get_label(self,file='train_label.txt'):
        """
        获取 model label
        :param file: the label filename,default in the ./data/
        :return:  label dict{id:age}
        """
       
        label = {}
        with open(os.path.join('./data/',file), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                arr = line.strip().split(',')
                t_id = arr[0]
                pro = arr[3]
                label[t_id] = pro 
        return label
        

    def make_data(self):

        train_label_dict=self.get_label('train_label.txt')
        test_label_dict=self.get_label('test_label.txt')

        train_data=[]
        self.train_label=[]

        test_data=[]
        self.test_label=[]

        with open('./data/tags.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                j_obj = json.loads(line)
                j_id = str(j_obj['id']) 
                j_tags = j_obj['tags']

                j_tags_str = ''
                for item in j_tags:
                    for key, val in item.items():
                        if key != 'weight':
                            j_tags_str += val + ' '

                if j_id in train_label_dict.keys():
                  
                    self.train_label.append(train_label_dict[j_id])
                    train_data.append(j_tags_str)

                if j_id in test_label_dict.keys():

                    self.test_label.append(test_label_dict[j_id])
                    test_data.append(j_tags_str)

        self.vectorizer = CountVectorizer()
        self.corpus_vec = self.vectorizer.fit(train_data+test_data)
        self.train_vec =  self.corpus_vec.transform(train_data)
        self.test_vec =  self.corpus_vec.transform(test_data)

        del train_data
        del test_data

    def train(self):
        
        print('start train model')
        
        self.clf = MultinomialNB()
        self.clf.fit(self.train_vec.toarray(), self.train_label)

        print('train model done')

    def eval(self):
        
        preds = self.clf.predict(self.test_vec.toarray())
        print('准确度:{0:.3f}'.format(metrics.accuracy_score(self.test_label, preds)))
        print('精度:{0:.3f}'.format(metrics.precision_score(self.test_label, preds, average='weighted')))
        print('召回:{0:0.3f}'.format(metrics.recall_score(self.test_label, preds, average='weighted')))
        print('f1-score:{0:.3f}'.format(metrics.f1_score(self.test_label, preds, average='weighted')))

    def predict(self,intro):
       
        intro_vec =  self.corpus_vec.transform([intro])
        preds = self.clf.predict(intro_vec.toarray())
        
        return preds
      

if __name__ == '__main__':

    my_model=Model()
    my_model.train()
    my_model.eval()


