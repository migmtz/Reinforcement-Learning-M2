import numpy as np
from matplotlib import pyplot as plt

datos = open("tmedata.txt", "r")

tableau = np.zeros((5000,15))

for i in range(0,5000):
    lineas = datos.readline()
    tableau[i,0:5] = (lineas.split(":")[1]).split(";")
    tableau[i,5:15] = (lineas.split(":")[2]).split(";")

datos.close()

## UCB

# Random

annonceur_rand = np.zeros((2,5000))

for i in range(0,5000):
    choisi = np.random.randint(1,11)
    annonceur_rand[0,i] = choisi
    annonceur_rand[1,i] = tableau[i,choisi+4]

print(sum(annonceur_rand[1,:]))

# Strategie StaticBest

annonceur_best = np.zeros((1,10))
max = 0
valmax = 0
for i in range(0,10):
    annonceur_best[0,i] = sum(tableau[:,5+i])
    if(annonceur_best[0,i] > valmax):
        max = i
        valmax = annonceur_best[0,i]

print(sum(tableau[:,5+max]))

# Strategie Optimale

annonceur_opt = np.zeros((2,5000))

for i in range(0,5000):
    valmax = 0
    maxi = 0
    for j in range(0,10):
        if(tableau[i,j+5] > valmax):
            maxi = j+1
            valmax = tableau[i,j+5]
    annonceur_opt[0,i] = maxi
    annonceur_opt[1,i] = valmax

print(sum(annonceur_opt[1,:]))

# UCB

annonceur_ucb = np.zeros((2,5000))
cumul = np.zeros((1,10))
passages = np.ones((1,10))

for i in range(0,10):
    annonceur_ucb[0,i] = i+1
    annonceur_ucb[1,i] = tableau[i,i+5]
    cumul[0,i] = annonceur_ucb[1,i]

for i in range(10,5000):
    aux = cumul/passages + np.sqrt(2*np.log(i+1)/passages)
    gagne = 0
    maxim = 0
    for j in range(0,10):
        if(aux[0,j] > maxim):
            gagne = j
            maxim = aux[0,j]
    annonceur_ucb[0,i] = gagne + 1
    annonceur_ucb[1,i] = tableau[i,gagne+5]
    cumul[0,gagne] += tableau[i,gagne+5]
    passages[0,gagne] += 1

print(sum(annonceur_ucb[1,:]))

plt.figure(1)
plt.subplot(121)
plt.plot(np.cumsum(annonceur_rand[1,:]),label="Random")
plt.plot(np.cumsum(tableau[:,5+max]),label="StaticBest")
plt.plot(np.cumsum(annonceur_opt[1,:]),label="Optimale")
plt.plot(np.cumsum(annonceur_ucb[1,:]),label="UCB")
plt.xlabel("t")
plt.ylabel("Gain cumulatif")
plt.legend()
plt.grid(True)
plt.show()

plt.subplot(122)
plt.plot(np.cumsum(annonceur_opt[1,:]-annonceur_rand[1,:]),label="Random")
plt.plot(np.cumsum(annonceur_opt[1,:]-tableau[:,5+max]),label="StaticBest")
plt.plot(np.cumsum(annonceur_opt[1,:]-annonceur_ucb[1,:]),label="UCB",c="r")
plt.legend()
plt.xlabel("t")
plt.ylabel("Regret")
plt.grid(True)
plt.show()

##Lin-UCB

Resultados = []

contextos = tableau[:,0:5]
recompensa = tableau[:,5:15]

alpha1 = 0.5
T = 5000

lista_ctxA = [np.identity(5,float) for i in range(0,10)]
lista_ctxB = [np.zeros((5,1),float) for i in range(0,10)]
lista_aux = [np.zeros((5,1),float) for i in range(0,10)]

alpha2 = [0.001,0.05,0.5,0.1,1,5,10]
for alpha in alpha2:
    Resultados = []
    for i in range(0,T):
        ctx_act = contextos[i]
        ctx_act = np.reshape(ctx_act,(5,1))
        maxl = 0
        argmaxl = 0
        for j in range(0,10):
            pt =  np.dot(np.transpose(lista_aux[j]),ctx_act) + alpha*np.sqrt(np.dot(np.dot(np.transpose(ctx_act),np.linalg.inv(lista_ctxA[j])),ctx_act))
            if pt > maxl :
                maxl = pt
                argmaxl=j
        Rec = recompensa[i,argmaxl]
        Resultados += [Rec]
        lista_ctxA[argmaxl] += np.dot(ctx_act,np.transpose(ctx_act))
        lista_ctxB[argmaxl] += Rec*ctx_act
        lista_aux[argmaxl] = np.dot(np.linalg.inv(lista_ctxA[argmaxl]),lista_ctxB[argmaxl])
    plt.figure(1)
    plt.plot(np.cumsum(Resultados),label="Lin-UCB avec alpha = %a"%alpha)
    plt.xlabel("t")
    plt.ylabel("Gain cumulatif")

plt.plot(np.cumsum(annonceur_opt[1,:]),label="Optimale")
plt.legend()
plt.grid(True)
plt.show()

# plt.figure(1)
# plt.subplot(121)
# plt.plot(np.cumsum(annonceur_rand[1,:]),label="Random")
# plt.plot(np.cumsum(tableau[:,5+max]),label="StaticBest")
# plt.plot(np.cumsum(annonceur_opt[1,:]),label="Optimale")
# plt.plot(np.cumsum(annonceur_ucb[1,:]),label="UCB")
# plt.plot(np.cumsum(Resultados),label="Lin-UCB avec 0.5",c="k")
# plt.xlabel("t")
# plt.ylabel("Gain cumulatif")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.subplot(122)
# plt.plot(np.cumsum(annonceur_opt[1,:]-annonceur_rand[1,:]),label="Random")
# plt.plot(np.cumsum(annonceur_opt[1,:]-tableau[:,5+max]),label="StaticBest")
# plt.plot(np.cumsum(annonceur_opt[1,:]-annonceur_ucb[1,:]),label="UCB",c="r")
# plt.plot(np.cumsum(annonceur_opt[1,:]-Resultados),label = "Lin-UCB avec 0.5",c="k")
# plt.legend()
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.grid(True)
# plt.show()
