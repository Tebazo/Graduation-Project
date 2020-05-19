import PyPDF2
import nltk
import math
from bs4 import BeautifulSoup
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from operator import itemgetter
import pandas as pd
import numpy as np
import unidecode
import re
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def pdf_to_text(pdfname):

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    
    #sio = BytesIO()
    codec = 'utf-8'
    laparams = LAParams()
    sio = StringIO()
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

def Stemming(sentence):
    stemmer = RSLPStemmer()
    phrase = []
    for word in sentence:
        if(len(word)>3):
            phrase.append(stemmer.stem(word.lower()))
    return phrase

def Stop_words(lista):
    a = [lista for lista in lista if lista not in stopwords]
    return a

def TF_calc(dic, tot_words):
    tf_dic={}
    for word, val in dic.items():
        tf_dic[word] = val/len(tot_words)
    return tf_dic

def IDF_calc(n_doc, lista):
    for i in range(len(lista)):
        lista_idf_final.append(math.log((n_doc/lista_idf[i]),base_log))
    return lista_idf_final

def Limpa_texto(texto):
    texto_sem_acentuacao = unidecode.unidecode(texto)
    texto_so_com_letras=re.sub(r'[^a-zA-Z ]+', ' ',texto_sem_acentuacao)
    return texto_so_com_letras.lower()

def Tokenizar_texto(texto_bruto):
    texto_limpo = Limpa_texto(texto_bruto)
    tokens = word_tokenize(texto_limpo, language = 'english')
    token_stopword = Stop_words(tokens)
    tokens_stemmer = Stemming(token_stopword)
    return tokens_stemmer
    #return token_stopword

def Dicionario_tokens(tokens):
    freq = {} 
    for item in tokens: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    return freq

def countTokens(text):
    tokens_count={}
    words=Tokenizar_texto(text)
    text=";".join(words)
    for i in range(len(words)):
        substring = words[i]
        if substring not in tokens_count:
            tokens_count[substring]=text.count(substring)
        for j in range (1,3):
            if(i+j > len(words)-1):
                break
            substring = substring+";"+ words[i+j]
            if substring not in tokens_count:
               tokens_count[substring]=text.count(substring)    
    return tokens_count

def countTokensFrase(text):
    tokens_count={}
    #text2=text
    words=Tokenizar_texto(text)
    if len(words) == 0:
        return 
    text=";".join(words)
    for i in range(len(words)):
        substring = words[i]
        if substring not in tokens_count:
           tokens_count[substring]=text.count(substring)
        for j in range (1,3):    #MUDANÇA DE 4 PARA 3
            if(i+j > len(words)-1):
                break
            substring = substring+";"+ words[i+j]
            if substring not in tokens_count:
               tokens_count[substring]=text.count(substring)  
    if len(tokens_count) == 0:
        #print('--',text,'##',text2,'$$')
        return
    dic_aux = sorted(tokens_count.items(), key=itemgetter(1), reverse=True)
    return dic_aux[0]



def countTokensSoma(text,x,y):
    words=text.split()
    words=Tokenizar_texto(text)
    text=";".join(words)
    if (x in text and y in text):
        return min(text.count(x),text.count(y))
    return 0
          

def countTokensCluster(text,g,x):
    words=text.split()
    words=Tokenizar_texto(text)
    text=";".join(words)
    somatorio = 0
    for i in g:
        if(i != x):
            somatorio += text.count(i)
    somatorio = somatorio * text.count(x)
    return somatorio


def neighbors(g,frase, dic_w):
    words=frase.split()
    words=Tokenizar_texto(frase)
    frase =";".join(words)
    produto = 0
    soma = 0
    for l in dic_w:
        for i in g:
            if(i != l):
                soma += frase.count(i)
        produto += soma * frase.count(l)
    return produto  



stopwords = stopwords.words('portuguese') + stopwords.words('english') + ["are","página","cite",'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for i in range(len(stopwords)):
    stopwords[i] = unidecode.unidecode(stopwords[i])



text = pdf_to_text("introducao.pdf")
texto = sent_tokenize(text)

dic_tokens = {}
for i in range(len(texto)):
    if texto[i].strip() != "":
        aux = countTokensFrase(texto[i])
        if aux:
            if(len(aux[0]) > 2):
                dic_tokens[aux[0]] = aux[1]
#3 abstracts artigos thiago, fabricio, carla, harlen


dic_ordenado = {}
dic_ord = sorted(dic_tokens.items(), key=itemgetter(1), reverse=True)
#MUDANÇA DE 30 PARA 10
if len(dic_ord)>50:
 tam = 50
 print('50')

else:
 tam = len(dic_ord)
 print(len(dic_ord))

for i in range(10):
    provisorio = dic_ord[i]
    dic_ordenado[provisorio[0]]= provisorio[1]



G=nx.Graph()
lista_com_keys = []
for i in dic_ordenado.keys():
    G.add_node(i, name=i)
    lista_com_keys.append(i)

dic_edges = {}
for i in range(len(lista_com_keys)-1):
    for j in range(i+1,len(lista_com_keys)):
        count = 0
        for l in range(len(texto)):
            count += countTokensSoma(texto[l],lista_com_keys[i],lista_com_keys[j])
        if (count > 0 ):
            G.add_edge(lista_com_keys[i],lista_com_keys[j])
            dic_edges[(lista_com_keys[i],lista_com_keys[j])] = count

Gfixo = G.copy()
arestas = list(G.edges())
Gnovo = G.copy()
for i in arestas:
    Gnovo.remove_edge(i[0], i[1])
    if not nx.has_path(Gnovo, i[0], i[1]):
        G.remove_edge(i[0], i[1])
        
    else:
        Gnovo.add_edge(i[0], i[1])

cluster = []
sub_graphs = list(nx.connected_component_subgraphs(G))
n = len(sub_graphs)
for i in range(n):
    cluster.append(sub_graphs[i].nodes())


dic_based = {}
for i in dic_tokens:
    for j in range(len(cluster)):
        based = 0
        for l in range(len(texto)):
            based += countTokensCluster(texto[l],cluster[j],i)
        dic_based[(i,j)] = based
        


dic_neighbors = {}
for i in range(len(cluster)):
    neighbors_count = 0
    for l in range(len(texto)):
        neighbors_count += neighbors(cluster[i],texto[l],dic_tokens)
    dic_neighbors[i] = neighbors_count


dic_key ={}
for i in dic_tokens:
    key = 1
    for g in range(len(cluster)):
        if(dic_neighbors[g] != 0):
            key *= (1-(dic_based[(i,g)]/dic_neighbors[g]))
    dic_key[i] = 1-key
    
dic_key = sorted(dic_key.items(), key=itemgetter(1), reverse=True)

dic_HK = {}
for i in range(15):
    provisorio = dic_key[i]
    dic_HK[provisorio[0]] = provisorio[1]

for i in dic_HK:
    if(i not in G.node()):
        Gnovo.add_node(i)

dic_collumn = {}
for i in dic_HK:
    for j in dic_ordenado:
        collumn = 0
        if(i != j):
            for l in range(len(texto)):
                collumn += countTokensSoma(texto[l],i,j)
        dic_collumn[(i,j)] = collumn

dic_final ={}
for i in dic_HK:
    for g in range(len(cluster)):
        controle = 0
        for j in cluster[g]:
            if(dic_collumn[(i,j)] > controle):
                controle = dic_collumn[(i,j)] 
                vertice = j
        if controle > 0:
            G.add_edge(i,vertice, collumn = controle)
            dic_final[i,vertice] = controle

Gfixo = G.copy()
arestas = list(Gnovo.edges())
for i in arestas:
    Gnovo.remove_edge(i[0], i[1])
    if not nx.has_path(Gnovo, i[0], i[1]):
        G.remove_edge(i[0], i[1])
        
    else:
        Gnovo.add_edge(i[0], i[1])
        
cluster2 = []
sub_graphs2 = list(nx.connected_component_subgraphs(G))
n = len(sub_graphs2)
for i in range(n):
    cluster2.append(sub_graphs2[i].nodes())

dic_top_collumn ={}
for i in list(G.nodes()):
    soma = 0
    for j in list(G.nodes()):
        aux = G.get_edge_data(i,j)
        if aux:
            soma += aux['collumn']
    dic_top_collumn[i] = soma

dic_top_collumn = sorted(dic_top_collumn.items(), key=itemgetter(1), reverse=True)

dic_finalzera = {}
for i in range(15):
    provisorio = dic_top_collumn[i]
    dic_finalzera[provisorio[0]] = provisorio[1]



#6719*