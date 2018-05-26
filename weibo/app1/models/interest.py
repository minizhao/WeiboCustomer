'''
Created on 2018年5月26日

@author: zhao
'''
import os
import jieba
import json
from sklearn.feature_extraction.text import CountVectorizer#文本特征提取
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics#模型评估
from blaze.expr.expressions import label

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
                interest = arr[4]
                label[t_id] = interest 
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
    
    def label_to_vec(self,intrs):
        
        self.label_value = ['体育','公益','其他','养生美容','写作','影视','摄影','旅游','时尚','明星','游戏','漫画','生活','电影','科技',\
                       '绘画','美容养生','美食','舞蹈','表演','运动','运动影视','阅读','阅读旅>游','音乐','音乐运动']
        arr=intrs.split("|")
        vec=[]
        for item in self.label_value:
            if item in arr:
                vec.append(1)
            else:
                vec.append(0)
        return vec

    def make_data(self):
   
        train_label_dict=self.get_label('train_label.txt')
        test_label_dict=self.get_label('test_label.txt')

        train_data=[]
        self.train_label=[]

        test_data=[]
        self.test_label=[]
        
        corpus={}  
              
        with open('./data/verifiedText.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                j_obj = json.loads(line)
                j_id = str(j_obj['id']) 
                if j_id in train_label_dict.keys():
                    desc_arr = self.cutWords(j_obj['verifiedText'].replace('\r\n', '').replace('\n', ''))                 
                    corpus[j_id]=" ".join(desc_arr)      
                if j_id in test_label_dict.keys():
                    desc_arr = self.cutWords(j_obj['verifiedText'].replace('\r\n', '').replace('\n', ''))                 
                    corpus[j_id]=" ".join(desc_arr)                  
        with open('./data/train_weibo.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                arr = line.strip().split('|')
                j_id = arr[0]
                desc_arr = self.cutWords(arr[1].replace('\r\n', ''))
                corpus[j_id] += ' '.join(desc_arr)         
        with open('./data/test_weibo.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                arr = line.strip().split('|')
                j_id = arr[0]
                desc_arr = self.cutWords(arr[1].replace('\r\n', ''))
                corpus[j_id] += ' '.join(desc_arr) 
                
        for key,value in train_label_dict.items():
            if key in corpus.keys():
                self.train_label.append(self.label_to_vec(value))
                train_data.append(corpus[key])
            
        for key,value in test_label_dict.items():
            if key in corpus.keys():
                self.test_label.append(self.label_to_vec(value))
                test_data.append(corpus[key])
                    
        self.vectorizer = CountVectorizer()
        self.corpus_vec = self.vectorizer.fit(train_data+test_data)
        self.train_vec =  self.corpus_vec.transform(train_data)
        self.test_vec =  self.corpus_vec.transform(test_data)

        del train_data
        del test_data
        
    def train(self):
        
        print('start train model')
        self.clf = RandomForestClassifier()
        self.clf.fit(self.train_vec.toarray(), self.train_label)
        print('train model done')
        
    def eval(self):

        preds = self.clf.predict(self.test_vec.toarray())
        correct = 0
        zh = 0
        total = len(self.test_label)*len(self.label_value)
        for idx_i,item in enumerate(self.test_label):
            for idx_j,icode in enumerate(item):
                if preds[idx_i][idx_j] == self.test_label[idx_i][idx_j]:
                    correct += 1
                if self.test_label[idx_i][idx_j] == 0:
                        zh += 1
                       
        print('准确度:{0:.3f}'.format(correct/total))
        print('召回:{0:.3f}'.format(zh/total))
        
    def predict(self,content):
        if type(content)==str:
            content=self.cutWords(content)
            content=[' '.join(content)]
        intro_vec =  self.corpus_vec.transform(content)
        preds = self.clf.predict(intro_vec.toarray())
        rst=[]
        for idx,value in enumerate(preds[0]):
            if value==1:
                rst.append(self.label_value[idx])
        print(preds)
        return ','.join(rst)
    
if __name__ == '__main__':

    my_model=Model()
    my_model.train()
    my_model.eval()
    
    