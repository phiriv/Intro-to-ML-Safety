from util import *
import numpy as np

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        #return self.output_fn(self.transition_fn(input_seq)) CREP
        state=self.start_state
        output=[]
        for inp in input_seq:
            state2=self.transition_fn(state, inp)
            output.append(self.output_fn(state2))
            return output
        


class Binary_Addition(SM):
    start_state = (0,0)

    def transition_fn(self, s, x):
        #return(s[-1]+x[-1]) #APPENDAGE
        (carry,digit)=s
        (i0,i1)=x
        tot=i0+i1+carry
        
        return 1 if tot>1 else 0,tot%2

    def output_fn(self, s):
        (carry,digit)=s
        return digit

class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s

class Reverser(SM):
    start_state = None

    def transition_fn(self, s, x):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.Wsx=Wsx
        self.Wsx=Wss
        self.Wsx=Wo
        self.Wss_0=Wss_0
        self.Wo_0=Wo_0
        self.f1=f1
        self.f2=f2
        self.l=self.Wsx.shape[1]
        self.m=self.Wss.shape[1]
        self.n=self.Wo.shape[1]
        self.start_state=np.zeros((self.n,1))
        

    def transition_fn(self, s, i):
        return self.f1(np.dot(self.Wss,s)+np.dot(self.Wsx,x)+self.Wss_0)
        

    def output_fn(self, s):
        return self.f2(np.dot(self.Wo,s)+self.Wo_0)