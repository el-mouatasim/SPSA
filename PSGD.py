import torch
import math
#import numpy as np



def euclidean_norm_squared(vec_list):
    return sum(torch.sum(v ** 2).item() for v in vec_list)

DENOMINATOR_EPS = 1e-25


class CLROptimizer:

    def __init__(self, net, criterion, c_inc=1.01, lr=0.01):
        self.lr = lr
        self.net = net
        self.criterion = criterion
        self.c_inc = c_inc # 1.05
        self.c_dec = 1/self.c_inc #0.95

    def step(self, X, y, loss):
        for f in self.net.parameters():
            f.data.sub_(self.lr*(f.grad.data))
            
        with torch.no_grad():
            grad_norm_squared = euclidean_norm_squared((-p.grad for p
                                                        in self.net.parameters()))
            learning_rate = self.lr
            lred = grad_norm_squared * learning_rate
            approx = loss.item() - lred
            
            y_hat, _ = self.net(X)    
            actual = self.criterion(y_hat, y).item()
            
            rel_err = (actual - approx) / (lred + DENOMINATOR_EPS)
            
        if rel_err > 0.5:
            h_mul = self.c_dec
        else:
            h_mul = self.c_inc
            
        self.lr *= h_mul
        return rel_err, grad_norm_squared
    
class CLROptimizer_pert:

    def __init__(self, net, criterion, lr=0.001):
        self.lr = lr
        self.net = net
        self.criterion = criterion
        self.kk = 1

    def lmbda(self):
        a = 0.001 #O.01 good
        lm = a /(math.sqrt(math.log(self.kk+2)))
        self.kk += 1
        return lm
    
    
    def step(self, X, y, loss, ksto=10,  save_name = 'parameters.pt'):
        
        actual = loss.item()
        torch.save(self.net.state_dict(), save_name)
        for f in self.net.parameters():
            f.data.sub_(self.lr*(f.grad.data))
        y_hat, _ = self.net(X)    
        w1 = self.criterion(y_hat, y).item()    
        if w1 < actual:
            torch.save(self.net.state_dict(), save_name)
            actual = w1        
        lm = self.lmbda()
        for k in range(ksto):
            self.net.load_state_dict(torch.load(save_name))    
            for f in self.net.parameters():
                pertu = torch.randn_like(f.data)
                f.data.sub_(self.lr*(f.grad.data)-lm*pertu)
                
            y_hat, _ = self.net(X)    
            w = self.criterion(y_hat, y).item()
            
            if w < actual:     
                torch.save(self.net.state_dict(), save_name)
                actual = w
    
        return 




