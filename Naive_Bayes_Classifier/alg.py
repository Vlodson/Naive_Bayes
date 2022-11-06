""" Naive Bayes za Iris DS 3 klase """

import pandas as pd
import numpy as np
from math import pi

# consts

class_probs = [0.33, 0.33, 0.33]
test_amount_const = 2

#=====================

df = pd.read_csv("C:\\Users\\vlada\\Desktop\\Datasets\\Iris DS\\Dataset.csv") # ucitan df
df_arr = df.to_numpy() # df u np array

# f za razdvajanje klasa
def sep_class(df_arr):

    sep = dict() # razdvojicu ih u dict jer je tako najbolje i najlakse za pratiti 

    for item in df_arr: # za svaki element dataframea
        item_class = item[-1] # ime klase je poslednji element u nizu jednog elementa
        item_atr = item[:-1] # atributi su svi sem poslednjeg u nizu

        if item_class not in sep: # ako ime klase nije vec vidjeno
            sep[item_class] = list() # pravim keyword u dict za taj broj

        sep[item_class].append(item_atr) # ako je vec vidjeno ime samo dodam na listu

    return sep

sep = sep_class(df_arr) # u sep su sada odvojene klase po imenima klasa

#=====================

# sredjivanje test slucajeva
test_cases = dict()

for x in sep:
    test_cases[x] = sep[x][:int(len(sep[x]) / test_amount_const)] # desetinu ukupnih stavljam u test cases
    del sep[x][:int(len(sep[x]) / test_amount_const)] # i tu desetinu brisem iz sep

# dodavanje labela
i = 0
labels = list()
for x in test_cases:
    for j in range(len(test_cases[x])):
        labels.append(i)
    i += 1

#=====================

# f koja pravi mean za svaki atribut klase i smesta u listu
def class_atr_mean(df_class):

    means = list() # lista u kojoj ce biti proseci

    for i in range(np.shape(df_class)[1]): # range atributa (ovde 4)

        j = 0 # restartovanje brojaca
        sum = 0 # restartovanje sume

        for j in range(np.shape(df_class)[0]): # range elemenata po klasi
            sum += df_class[j][i] # sumiram po elementu po atributu

        means.append(sum/np.shape(df_class)[0]) # prosek je suma kroz br clanova

    return means

# smestam sve proseke u means listu svih proseka svih klasa
all_means = list()
for x in sep: # ovo ce mi dati svaki keyword iz dict
    all_means.append(class_atr_mean(sep[x])) # na means dodajem f od gore po svakom keywordu tj klasi

#=====================

# f za sve std dev po klasi
def class_atr_std_dev(df_class, class_means):

    std_dev = list() # lista u kojoj ce biti devijacije

    for i in range(np.shape(df_class)[1]): # range atributa (ovde 4)

        j = 0 # restartovanje brojaca
        sum = 0 # restartovanje sume

        for j in range(np.shape(df_class)[0]): # range elemenata po klasi
            sum += (df_class[j][i] - class_means[i])**2 # sumiram po elementu po atributu

        std_dev.append( (sum/np.shape(df_class)[0])**0.5 ) # prosek je suma kroz br clanova pa koren

    return std_dev

# na istu foru kao i gore samo sada koristim i ctr za proseke jer nemam brojac nego elementwise gledam
all_std_dev = list()
ctr = 0
for x in sep:
    all_std_dev.append(class_atr_std_dev(sep[x], all_means[ctr]))
    ctr += 1

#=====================

# ulaz ti je ono sto treba da klasifikujes oblika [a1, a2, ...]

# gausova f vrv za jednu klasu, samo po formuli i vrv po atr stavlja u listu
def class_gauss(x, mean, std_dev):

    probs = list()

    for i in range(len(x)):
        a = 1 / (std_dev[i] * (2*np.pi)**0.5)
        b = np.exp( -1 * 0.5 * ( (x[i] - mean[i]) / std_dev[i])**2 )
        temp = a * b
        probs.append(temp)
    
    return probs

#=====================

def clf(x, num_of_classes, all_means, all_std_dev, class_probs):

    res = list()

    for i in range(num_of_classes):
        probs = class_gauss(x, all_means[i], all_std_dev[i])
        temp = 1
        for p in probs:
            temp *= p
        
        res.append(temp*class_probs[i])

    return res.index(max(res))

#=====================

# testiranje
pred = list()
for x in test_cases:

    for i in range(len(test_cases[x])):
        pred.append(clf(test_cases[x][i], len(sep), all_means, all_std_dev, class_probs)) # predikcije stavljam u listu posebnu pa cu da uporedjujem

# odredjivanje tacnosti (verovatno moze brze)
ctr = 0
for i in range(len(labels)): # samo uvecavam brojac za jedan svaki put kada se poklopi label i pred
    if labels[i] == pred[i]:
        ctr += 1

print(pred)
print(labels)
print('Tacnost je {}%'.format(ctr / len(labels) * 100))