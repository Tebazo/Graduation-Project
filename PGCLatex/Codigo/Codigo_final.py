import re
import os
import pandas as pd
from unidecode import unidecode
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import RSLPStemmer
from string import punctuation
from nltk.probability import FreqDist
#from io import BytesIO
#from cStringIO import StringIO
from io import StringIO
from collections import OrderedDict
import math
from operator import itemgetter
import numpy as np
from scipy import spatial


        
def pdf_to_text(pdfname):
    print("Transformando PDF em txt ...")
    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    #sio = BytesIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Extract text
    fp = open(pdfname, 'rb')
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp,pagenos, 
                                      maxpages=maxpages,
                                      password=password,
                                      caching=caching,
                                      check_extractable=False):
        interpreter.process_page(page)
    fp.close()
    # Get text from StringIO
    text = sio.getvalue()
    # Cleanup
    device.close()
    sio.close()
    return text

def limpaTexto(text):
    print("Retirando caracteres especiais ...")
    #retirando toda a acentuacao 
    text = unidecode(text)
    text=re.sub(r'[\~\ç\´\'\,\`\^]+','',text)
    #pegando somente tokens formados por caracteres de a ate z
    text=re.sub(r'[^a-zA-Z]+',' ',text)
    return text

def geraTokens(text):
    print("Gerando tokens do texto ...")
    #tokenizando o text
    palavras = word_tokenize(text, language='portuguese')
    #transformando todos os tokens em minusculo
    palavras = [palavra.lower() for palavra in palavras]
    return palavras

def passaStopwords(palavras):
    print("Retirando stopwords ...")
    from nltk.corpus import stopwords
    #pegando uma lista de stopword da lingua portuguesa
    stopwords=stopwords.words("portuguese")
    #retirando a acentuacao de todas as palavras do stopword
    for i in range(len(stopwords)):
        stopwords[i]=unidecode(stopwords[i])
    #filtrando as palavras existentes na lista de stopwords
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords]
    return palavras_sem_stopwords

def Stemming(sentence):
    print("Gerando stemmers ...")
    stemmer = RSLPStemmer()
    phrase = []
    for word in sentence:
        if(len(word)>3):
            phrase.append(stemmer.stem(word.lower()))
    return phrase


def geraDicPDF(palavras_com_stemmer):
    print("Criando o dicionario do PDF ...")
    #calculando a frequencia de cada palavra no texto
    fdist = FreqDist(palavras_com_stemmer)
    
    dicPDF=dict(fdist)
    dicPDF=OrderedDict(sorted(dicPDF.items()))
    return dicPDF

def atualizaDicwords(dicPDF):
    global dic_words
    for key, value in dicPDF.items():
        if key in dic_words:
           dic_words[key]+=value
        else:
           dic_words[key]=value 
           

def palavrasProjeto(dicPDF):
    dic_wproj = {}
    for key, value in dicPDF.items():
        if key in dic_wproj:
           dic_wproj[key]+=value
        else:
           dic_wproj[key]=value 
    return dic_wproj
           
def lendo_arqs(working_dir):
    for root,dirs,files in os.walk(working_dir):
        print("============= Lendo PDFs ==================")
    for name in files:
      f= os.path.join(root, name) 
      print("Processando arquivo: ",f )
      try:
        global files_pdfs
        files_pdfs.append(name)
        text=pdf_to_text(f)
        text=limpaTexto(text)
        palavras=geraTokens(text)
        palavras_sem_stopwords=passaStopwords(palavras)
        palavras_com_stemmer=Stemming(palavras_sem_stopwords)
        dicPDF=geraDicPDF(palavras_com_stemmer)
        atualizaDicwords(dicPDF)
        global dic_geral
        dic_geral[name]=dicPDF
      except:
        print("Impossivel abrir o pdf ",f)
        
def calculaTF_Proj(dic):
    dic_proj_tf={}
    for k,v in dic.items():
        dic_proj_tf[k] = v/sum(dic.values())
    return dic_proj_tf
        
def calculaTF_Geral():
    global dic_geral
    dic_geral_tf={}
    for k,v in dic_geral.items():
        dic_tf={}
        for key, value in v.items():
            dic_tf[key] = value/sum(v.values())
        dic_geral_tf[k] = dic_tf
    return dic_geral_tf

def calculaIDF():
    global dic_geral
    global dic_words
    global files_pdfs
    dic_idf ={}
    for k,v in dic_words.items():
        count=0
        for k_docs, v_docs in dic_geral.items():
            if k in v_docs:
                count+=1          
        dic_idf[k]=math.log10(len(files_pdfs)/count)
    return dic_idf

def calculaTFIDF(dic_geral_tf,dic_idf):
    global dic_geral
    dic_tfidf={}
    for k,v in dic_geral_tf.items():
        dic_temp={}
        for key,value in v.items():
            dic_temp[key] = value * dic_idf[key]
        dic_tfidf[k] = dic_temp
    return dic_tfidf

def calculaTFIDF_Proj(dic_proj,dic_idf):
    dic_tfidf_proj = {}
    for k,v in dic_proj.items():
        if k in dic_idf:
            dic_tfidf_proj[k] = v * dic_idf[k]
    return dic_tfidf_proj
            
def calculaSimilaridade(dic_wproj):
    global dic_geral
    dic_similaridade = {}
    dic_wproj = OrderedDict(sorted(dic_wproj.items()))
    for k,v in dic_geral.items():
        lista_temp = []
        for key,value in dic_wproj.items():
            if key in v:
                lista_temp.append(v[key])              
            else:
                lista_temp.append(0.0)
        result = 1-spatial.distance.cosine(lista_temp,list(dic_wproj.values()))
        dic_similaridade[k] = result
        #dic_similaridade = sorted(dic_similaridade.items(), key=itemgetter(1), reverse=True)
    return dic_similaridade
                
            

def lendo_proj(proj_dir,dic_idf):
    dic_final = {}
    for rr,dd,ff in os.walk(proj_dir, topdown = True):
        print("============= Lendo PDFs ==================")
    for name in ff:
       f= os.path.join(rr, name)
       print("Processando arquivo:",f)
       try:
          novo = {}
          global files_proj
          files_proj.append(name)
          text=pdf_to_text(f)
          text=limpaTexto(text)
          palavras=geraTokens(text)
          palavras_sem_stopwords=passaStopwords(palavras)
          palavras_com_stemmer=Stemming(palavras_sem_stopwords)
          dicPDF=geraDicPDF(palavras_com_stemmer)
          dic_wproj = palavrasProjeto(dicPDF)      
          dic_tf_proj = calculaTF_Proj(dic_wproj)
          #dic_tfidf_proj = calculaTFIDF_Proj(dic_tf_proj,dic_idf)
          #dic_final[name] = calculaSimilaridade(dic_tfidf_proj)
          dic_teste = sorted(dic_tf_proj.items(), key=itemgetter(1), reverse=True)
          for i in range(15):
            provisorio = dic_teste[i]
            novo[provisorio[0]]= provisorio[1]
          dic_tfidf_proj = calculaTFIDF_Proj(novo,dic_idf)
          dic_final[name] = calculaSimilaridade(dic_tfidf_proj)


    

       except:
          print("Impossivel abrir o pdf:",f)
    return dic_final

    

files_proj = []
files_pdfs=[]
files_proj=[]
dic_words={}      
dic_geral={}
proj_dir ="./Projeto"
working_dir="./PDF"
lendo_arqs(working_dir)
#ordenando o dicionario dic_geral
dic_words=OrderedDict(sorted(dic_words.items()))
#criando o DataFrame bag_of_words
df_bow = pd.DataFrame(index=files_pdfs, columns=dic_words.keys())
df_bow = df_bow.fillna(0)
#populando o dataFrame
#print("Aguarde, criando o DataFrame BoW")
#for f in files_pdfs:
 #   print("inserindo o PDF: ",f)
  #  freq_words=dic_geral[f]
   # for key,value in freq_words.items():
    #    df_bow.loc[f,key]=value    

dic_geral_tf = calculaTF_Geral()
dic_idf = calculaIDF()
dic_tfidf = calculaTFIDF(dic_geral_tf,dic_idf)
dic_final=lendo_proj(proj_dir,dic_idf)




