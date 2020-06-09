import numpy as np
import matplotlib.pyplot as plt 
# from epsilon_greedy_bandit.py import run_exp as epsilon_greedy_exp

class Bandits :
    def __init__(self, m , upper_limit ):
        self.m = m 
        self.mean = upper_limit
        self.N = 0

    def choose(self) :
        return np.random.randn() + self.m     

    def update(self , x) :
        self.N =+ 1
        self.mean = ( 1-1.0 / self.N )*self.mean + (1/self.N)*x

#oiv
def run_exp ( m1 , m2 , m3 , N , upper_limit=10  ) :

    bandits = [ Bandits(m1 , upper_limit) , Bandits(m2 , upper_limit) , Bandits(m3 , upper_limit) ]

    data = np.empty(N) 
    for i in range(N): 
        # epsilon greedy 
        # p = np.random.random() 
        # if p < eps: 
        #     j = np.random.choice(3) 
        # else: 
        j = np.argmax([a.mean for a in bandits]) 
        x = bandits[j].choose() 
        bandits[j].update(x) 
        
        # for the plot 
        data[i] = x 

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1) 
    
    # plot moving average ctr 
    plt.plot(cumulative_average) 
    plt.plot(np.ones(N)*m1) 
    plt.plot(np.ones(N)*m2) 
    plt.plot(np.ones(N)*m3) 
    plt.xscale('log') 
    plt.show() 
    
    for a in bandits: 
        print(a.mean) 
    
    return cumulative_average 
  

  # e-greedy algorithm 
def epsilon_greedy_exp ( m1 , m2 , m3 , eps , N ) :

    bandits = [ Bandits(m1 , 10) , Bandits(m2 , 10) , Bandits(m3 , 10) ]

    data = np.empty(N) 
    for i in range(N): 
        # epsilon greedy 
        p = np.random.random() 
        if p < eps: 
            j = np.random.choice(3) 
        else: 
            j = np.argmax([a.mean for a in bandits]) 
        x = bandits[j].choose() 
        bandits[j].update(x) 
        
        # for the plot 
        data[i] = x 
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)   
    return cumulative_average 



if __name__ == '__main__': 
      
    c_1 = epsilon_greedy_exp(1.0, 2.0, 3.0, 0.009, 100000)
    oiv = run_exp(1.0, 2.0, 3.0, 100000)


# log scale plot 
plt.plot(c_1, label ='eps = 0.01') 
plt.plot(oiv, label ='oiv') 
plt.legend() 
plt.xscale('log') 
plt.show()


  # linear plot 
plt.figure(figsize = (12, 8)) 
plt.plot(c_1, label ='eps = 0.01') 
plt.plot(oiv, label ='oiv') 
plt.legend() 
plt.show()   