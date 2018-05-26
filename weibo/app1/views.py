from django.shortcuts import render
from django.http import HttpResponse
from .models.age import Model as agemodel
from .models.gender import Model as genmodel
from .models.interest import Model as intrsmodel
import json
import os
import pickle as pkl



def predict(request):
    
    if os.path.isfile(os.path.join('saves','age_m.pkl')):
        age_m=pkl.load(open(os.path.join('saves','age_m.pkl'),'rb'))
        
    else:
        age_m=agemodel()
        age_m.train()
        pkl.dump(age_m,open(os.path.join('saves','age_m.pkl'),'wb'))
        
        
        
    if os.path.isfile(os.path.join('saves','intrs_m.pkl')):
        intrs_m=pkl.load(open(os.path.join('saves','intrs_m.pkl'),'rb'))
        
    else:
        intrs_m=intrsmodel()
        intrs_m.train()
        pkl.dump(intrs_m,open(os.path.join('saves','intrs_m.pkl'),'wb'))
        
        
        
    
    if os.path.isfile(os.path.join('saves','gen_m.pkl')):
        gen_m=pkl.load(open(os.path.join('saves','gen_m.pkl'),'rb'))
        
    else:
        gen_m=genmodel()
        gen_m.train()
        pkl.dump(gen_m,open(os.path.join('saves','gen_m.pkl'),'wb'))
         
        

    
    
    intro=request.POST.get('introduction')
    verf=request.POST.get('verifiedText')
    wcontent=request.POST.get('weiboContent')
    
    age_pred=age_m.predict(intro)
    gen_pred=gen_m.predict(verf)
    intrs_pred=intrs_m.predict(verf+wcontent)
    
    dict_={}
    dict_['age:']=age_pred[0]
    dict_['gender:']=gen_pred[0]
    dict_['interest:']=intrs_pred
    
    return HttpResponse(json.dumps(dict_))


def home(request):
    
    return render(request, 'index.html')