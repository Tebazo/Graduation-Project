import nltk
import math
from bs4 import BeautifulSoup
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from operator import itemgetter
import pandas as pd
import numpy as np
import unidecode
import re
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity

def pdf_to_text(pdfname):

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
    texto_so_com_letras=re.sub(r'[^a-zA-Z]+',' ',texto_sem_acentuacao)
    return texto_so_com_letras.lower()

def Tokenizar_texto(texto_bruto):
    texto_limpo = Limpa_texto(texto_bruto)
    tokens = word_tokenize(texto_limpo, language = 'portuguese')
    token_stopword = Stop_words(tokens)
    tokens_stemmer = Stemming(token_stopword)
    return tokens_stemmer

def Dicionario_tokens(tokens):
    freq = {} 
    for item in tokens: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    return freq

stopwords = stopwords.words('portuguese') + stopwords.words('english') + ['pagina','cite''a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for i in range(len(stopwords)):
    stopwords[i] = unidecode.unidecode(stopwords[i])

text = pdf_to_text("Fantinato_DenisGustavo_M.pdf")
#text.strip()
text= Limpa_texto(text)
#text = re.sub('\n','',text)
file1 = open("myfile.txt","w")
file1.write(text)
file1.close()

token_desc = Tokenizar_texto(desc)
lista_desc.append(token_desc)
dic_docente = TF_calc(Dicionario_tokens(token_desc),token_desc)
lista_tf_docente.append(dic_docente)
lista_lista.append(token_desc)
 



tokens_geral += Tokenizar_texto(desc_geral)
dic_geral += Dicionario_tokens(tokens_geral)


tokens_projeto= Tokenizar_texto(text)
dic_proj = Dicionario_tokens(tokens_projeto)


################################################################3333
#Tratamento do xml começa aqui

with open("3nomes.xml","r") as f:
    file = f.read()
f.close()
#infile =  open("3nomes.xml","r")
#contents = infile.read()
soup = BeautifulSoup(file,'xml')
pesquisadores = soup('pesquisador')
lista_desc = []
lista_idf = []
lista_tf_docente = []
lista_idf_final = []
lista_comp = []
dic_comp = {}
dic_tfidf_proj = {}
base_log = 10



lista_lista = []
desc_geral = " "
for pesquisador in pesquisadores:
    projeto_pesquisa = pesquisador("projetos_pesquisa")
    desc = " "
   

    titulos = pesquisador("titulo")
    revistas = pesquisador("revista")
    for titulo in titulos:
        desc += " " + unidecode.unidecode(titulo.get_text())
        desc_geral += " " + unidecode.unidecode(titulo.get_text())
    for revista in revistas:
        desc += " " + unidecode.unidecode(revista.get_text())
        desc_geral += " " + unidecode.unidecode(revista.get_text())

    for i in projeto_pesquisa:
        descricoes = i("descricao")
        nomes = i("nome")
        
        for nome in nomes:
            desc += " " + unidecode.unidecode(nome.get_text())
            desc_geral += " " + unidecode.unidecode(nome.get_text())
        
        for descricao in descricoes:
            desc += " " + unidecode.unidecode(descricao.get_text())
            desc_geral += " " + unidecode.unidecode(descricao.get_text())
    token_desc = Tokenizar_texto(desc)
    lista_desc.append(token_desc)
    dic_docente = TF_calc(Dicionario_tokens(token_desc),token_desc)
    lista_tf_docente.append(dic_docente)
    lista_lista.append(token_desc)
 



tokens_geral = Tokenizar_texto(desc_geral)
dic_geral = Dicionario_tokens(tokens_geral)

df = pd.DataFrame(0, index=np.arange(len(lista_desc)), columns=dic_geral)
df_tf = pd.DataFrame(0.0, index=np.arange(len(lista_desc)), columns=dic_geral)
df_docente_tfidf = pd.DataFrame(0.0, index=np.arange(len(lista_desc)), columns=dic_geral)
#rotina para popular o dataframe
for i in range(len(lista_desc)):
    tokens=lista_desc[i]
    df.at[i, tokens]=1

for i in range(len(dic_geral)):
    lista_idf.append(df.iloc[:,i].sum())

lista_com_idf = IDF_calc(len(lista_desc),lista_idf)
dic_com_idf= {}

#verificar se o count pode ser utilizado!!!!!!!!!!!!!!!
count = 0
for i in dic_geral.keys():
    dic_com_idf[i] = lista_com_idf[count]
    count +=1

for i in range(len(lista_tf_docente)):
    aa = lista_tf_docente[i]
    for n in aa.items():
        df_tf.at[i, n[0]] = n[1]
#verificado!!!!!
for i in range(len(dic_geral)):
    df_docente_tfidf.iloc[:,i] = (df_tf.iloc[:,i]*lista_com_idf[i])
    

############################### verificado!!!!!

for i in dic_proj.keys():
    for j in dic_geral.keys():
        if(i == j):
            lista_comp.append(i)
            dic_comp[i] = dic_proj[i]


############################## ALTERADO###########################
"""
for i in dic_finalzera.keys():
    for j in dic_geral.keys():
        if(i == j):
            lista_comp.append(i)
            dic_comp[i] = dic_proj[i]
"""
dic_tf_proj = TF_calc(dic_comp, tokens_projeto)


for i in dic_com_idf.keys():
    if(i in dic_tf_proj):
        dic_tfidf_proj[i] = dic_tf_proj[i]*dic_com_idf[i]
    else:
        dic_tfidf_proj[i] = 0.0


df_similaridade = pd.DataFrame(0.0, index=np.arange(len(lista_desc)),columns = ['Similaridade'])


for i in dic_com_idf.keys():
    if(i in dic_tf_proj):
        dic_tfidf_proj[i] = dic_tf_proj[i]*dic_com_idf[i]
    else:
        dic_tfidf_proj[i] = 0.0
lista_proj_tfidf = []
#Gerando lista com valores do dic_tfidf_proj, ordenando de acordo com as chave do
#dic_geral, pois ele que ordena o dataframe
for i in dic_geral.keys():
    lista_proj_tfidf.append(dic_tfidf_proj[i])
    
lista_proj_tfidf_array = np.array([lista_proj_tfidf])


for i in range(len(lista_desc)):
    #lista temporaria, armazena os valores das linhas do df_docente_tfidf
    lista_temporaria = []
    #lista_temporaria.clear()
    for j in dic_geral.keys():
        lista_temporaria.append(df_docente_tfidf.at[i,j].item())
    lista_temporaria = np.array([lista_temporaria])
    df_similaridade.at[i,'Similaridade'] = cosine_similarity( lista_temporaria, lista_proj_tfidf_array)


#data=pd.read_csv('../datasets/cetesb_dinamica_2019.csv',header=0, )
#data.to_csv(arq, sep=',',encoding='utf-8')
#arq="../datasets/"+str(ano)+".csv"
#data.to_csv("c:/mario/tdidfdocs.csv", sep=",", enconding="utf-8")
#Luhn


