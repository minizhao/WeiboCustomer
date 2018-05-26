'''
Created on 2018年5月26日

@author: zhao
'''
import os
import json
from sklearn.feature_extraction.text import CountVectorizer#文本特征提取
from sklearn.linear_model import LogisticRegression#逻辑回归
from sklearn import metrics#模型评估
import jieba


class Model(object):
   
    def __init__(self):
        super(Model, self).__init__()
        self.stopWords = self.loadStopWords()
        self.make_data()
        
    def loadStopWords(self):
        """
        获取停用词表
        :param stopWords list
        :return: 分词后list
        """
        stopw =[line.strip() for line in open('./data/stopWords.txt', 'r',encoding='utf-8').readlines()]
        return stopw
    
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
                gender = arr[1]
                label[t_id] = gender 
        return label

    def cutWords(self,msg):
        """
        分词
        :param context:待分词的文本
        :return: 分词后的文本
        """
        seg_list = jieba.cut(msg,cut_all=False)  
        #key_list = jieba.analyse.extract_tags(msg,20) #get keywords   
        leftWords = []   
        for i in seg_list:  
            if (i not in self.stopWords):  
                leftWords.append(i)          
        return leftWords  
    
    def make_data(self):

        train_label_dict=self.get_label('train_label.txt')
        test_label_dict=self.get_label('test_label.txt')

        train_data=[]
        self.train_label=[]

        test_data=[]
        self.test_label=[]
        
        with open('./data/verifiedText.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                j_obj = json.loads(line)
                j_id = str(j_obj['id']) 

                if j_id in train_label_dict.keys():
                    desc_arr = self.cutWords(j_obj['verifiedText'].replace('\r\n', '').replace('\n', ''))
                    self.train_label.append(train_label_dict[j_id])
                    train_data.append(' '.join(desc_arr))

                if j_id in test_label_dict.keys():
                    desc_arr = self.cutWords(j_obj['verifiedText'].replace('\r\n', '').replace('\n', ''))
                    self.test_label.append(test_label_dict[j_id])
                    test_data.append(' '.join(desc_arr))

        self.vectorizer = CountVectorizer()
        self.corpus_vec = self.vectorizer.fit(train_data+test_data)
        self.train_vec =  self.corpus_vec.transform(train_data)
        self.test_vec =  self.corpus_vec.transform(test_data)


        del train_data
        del test_data
        
    def train(self):
        
        self.clf = LogisticRegression()
        self.clf.fit(self.train_vec.toarray(), self.train_label)

        print('train model done')
        
    def eval(self):

        preds = self.clf.predict(self.test_vec.toarray())

        print('准确度:{0:.3f}'.format(metrics.accuracy_score(self.test_label, preds)))
        print('精度:{0:.3f}'.format(metrics.precision_score(self.test_label, preds, average='weighted')))
        print('召回:{0:0.3f}'.format(metrics.recall_score(self.test_label, preds, average='weighted')))
        print('f1-score:{0:.3f}'.format(metrics.f1_score(self.test_label, preds, average='weighted')))
        
    def predict(self,intro):
        if type(intro)==str:
            intro=self.cutWords(intro)
            intro=[' '.join(intro)]
        intro_vec =  self.corpus_vec.transform(intro)
        preds = self.clf.predict(intro_vec.toarray())
        
        return preds
      

if __name__ == '__main__':

    my_model=Model()
    my_model.train()
    my_model.eval()



        
        