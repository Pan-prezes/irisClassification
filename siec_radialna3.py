import numpy as np
import iris
import matplotlib.pyplot as plt

#funkcja obliczająca wagi w2 i bias2, rozwiązuje rówanie macierzowe Ax=b 
def solvelin3(p,t):
    x=np.shape(p)[0]
    y=np.shape(p)[1]
    p=np.vstack([p,np.ones((1,y))])

    c = np.dot(t, np.linalg.pinv(p)) #rozwiązywanie równania W=Gplus d macież z pseudoinwersją
    return [c[:,0:x],c[:,[x]]]
    

#funkcja obliczająca dystanse pomiędzy poszczególnymi wektorami
def dist(x,y):
    sum = np.zeros((np.shape(x)[0],np.shape(y)[0]))
    for i in range(0, np.shape(x)[0]):
        for j in range(0, np.shape(y)[0]):
            sum[i][j]=np.linalg.norm(np.transpose(x[i])-np.transpose(y[j]))
    return sum

#funkcja realizująca funkcję Gaussa exp(-n*n^2)
def radbas(n):
    return np.exp(-(n*n))

#funkcja sumująca w osi Y
def sum(n):
    return n.sum(axis=1)


class RBF:
    def __init__(self, Ps, Ts, sv, n):
        self.Ps = Ps
        self.Ts = Ts
        self.sv = sv
        self.N = n
        self.R = np.shape(Ps)[1]
        self.Q = np.shape(Ps)[0]
        self.b = (np.sqrt(-np.log(0.5)))/sv #zastosowano te same obliczenia co w MATLAB R2020a
                                            #bias po podniesieniu do kwadratu odpowiada 1/(2sigma^2)
    
        self.w1=[]
        self.w2=[]
        self.b2=[]

    
    def learn(self):
        plotx=[]
        ploty=[]

        P = radbas(dist(self.Ps, self.Ps)*self.b) 
        PP = np.transpose(sum(P*P)) 
        PP=PP[:,np.newaxis]
        d = self.Ts
        dd = np.transpose(np.sum(d*d,axis=0)) 
        dd=dd[:,np.newaxis]
        e = (np.dot(P.T,d).T*np.dot(P.T,d).T)/np.multiply(dd,PP.T) #tworzenie macierzy na podstawie której wyszukiwany maksymalny błąd
        
        pick = np.argmax(np.sum(e*e,axis=0)) #wybieranie indeksu z największym błędem
        used = np.array([pick]) 
        left = np.arange(0,self.Q)
        W = P[:,pick]
        W=W[:,np.newaxis]
        P = np.delete(P,pick,axis=1) #usunięcie kolumny z największym błędem
        
        PP = np.delete(PP,pick,axis=0)
        
        e = np.delete(e,pick,axis=1)
        left = np.delete(left,pick,0)
        
        
        w1 = self.Ps[used, :] #wybranie pierwszego centrum
        a1 = radbas(dist(w1, self.Ps)*self.b) #obliczanie macierzy Greena
        [w2,b2]=solvelin3(a1,np.transpose(self.Ts)) #obliczenie wag i biasu
        a2 = w2*a1+ b2*np.ones(self.Q) #przemnożenie macierzy przez wagi i dodanie biasu /symulacja sieci
        MSE = np.square(a2 - self.Ts.T).mean() #obliczenie błędu MSE
        print(MSE)
    
        plotx.append(1)
        ploty.append(MSE)
        
        plt.plot(plotx,ploty)
        
        for i in range(1,self.N):   #w pętli for dzieje się mniej wiecej to samo co powyżej w funkcji

            wj = W[:,i-1]
            wj=wj[:,np.newaxis]

            a = np.dot(wj.T,P)/np.dot(wj.T,wj)

            P = P - np.dot(wj,a)
            PP = np.transpose(np.sum(P*P,axis=0))


            e = (np.dot(P.T,d).T*np.dot(P.T,d).T)/np.multiply(dd,PP.T)#tworzenie macierzy na podstawie której wyszukiwany maksymalny błąd
            pick = np.argmax(np.sum(e*e,axis=0))#wybieranie indeksu z największym błędem
            used=np.hstack([used,left[pick]])
            W = np.hstack([W,P[:,[pick]]])
            P = np.delete(P,pick,1)
            PP = np.delete(PP,pick,0)
            e = np.delete(e,pick,1)
            left = np.delete(left,pick,0)
            w1 = self.Ps[used, :] #wybranie centrów.
            a1 = radbas(dist(w1, self.Ps)*self.b)#obliczanie macierzy greena
            [w2,b2]=solvelin3(a1,np.transpose(self.Ts))#obliczenie wag i biasu
            a2 = np.dot(w2,a1) + b2*np.ones(self.Q)#przemnożenie macierzy przez wagi i dodanie biasu /symulacja sieci
            MSE = np.square(a2 - self.Ts.T).mean() #obliczenie błędu MSE
            print(MSE)
            plotx.append(i+1)
            ploty.append(MSE)
        
        plt.plot(plotx,ploty)
        plt.xlabel('Ilość neuronów')
        plt.ylabel('MSE')
        plt.show()

            
        self.w1=w1
        self.w2=w2
        self.b2=b2
        
    def sim(self,Ps):#funkcja symulująca sieć
        a1 = radbas(dist(self.w1, Ps)*self.b) #utworzenie macierzy Greena
        a2 = np.dot(self.w2,a1) + self.b2*np.ones(np.shape(Ps)[0]) #przemnożenie macierzy greena przez wagi i dodanie biasu
        return a2.T
            
           

        

        
Pns=np.array(iris.Pns)
Ts=np.array(iris.TTs)

net = RBF(Pns,Ts,0.2,70)
net.learn()

print(np.around(net.sim(Pns)))
