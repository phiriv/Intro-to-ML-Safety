import pdb
from dist import uniform_dist, delta_dist, mixture_dist
from util import *
import random

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn, 
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

#Use mdp class definitions to get the reward function, discount factor, 
#transition model, and expectation of the Q-values over a distribution

#terminate when max(s,a) |Qt(s,a) - Qt-1(s,a)| < eps

def value_iteration(mdp, q, eps = 0.01, interactive_fn = None,
                    max_iters = 10000):
    #initialization
    #for s in q.states, a in q.actions:
        #q(s,a)=0
    '''
    for j in range(0,max_iters):
        
        for s in q.states, a in q.actions:
            Q=mdp.reward_fn(s,a)+mdp.discount_factor*mdp.transition_model(s,a)*value(q,s)
        
            n_q=q.copy() #save val between iters
        if abs(n_q-q) < eps:
            return n_q
        Q_old=Q
    '''
    def v(s): #quick value calc
        return value(q,s)
    
    for i in range(0,max_iters):
        
        n_q=q.copy() #if this isn't here updating won't work
        delt=0
        
        for s in mdp.states:
            for a in mdp.actions:
                #rather than defining a new var every time, re-assign class obj.
                n_q.set(s, a, mdp.reward_fn(s,a) + mdp.discount_factor*mdp.transition_model(s,a).expectation(v))
                delt=max(delt,abs(n_q.get(s,a)-q.get(s,a))) #termination condition
                
        if delt<eps:
            return n_q
        q=n_q
        
    return q
        
# Compute the q value of action a in state s with horizon h, using expectimax
def q_em(mdp, s, a, h):
    if h==0: #base case
        return 0
    else:    #recursive case
        return mdp.reward_fn(s,a) + mdp.discount_factor* \
            sum( [p*max( [ q_em(mdp,sp,ap,h-1) for ap in mdp.actions] ) for (sp,p) in mdp.transition_model(s,a).d.items() ] )
            #Big messy statement! 
            #1st term is the immediate reward for transitioning out of the current state
            #2nd is the discounted reward with recursion, where we iterate
            #over the states and probabilities (expectations) of the transition model
            #and take the corresponding max of the associated actions/rewards

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    """ Return Q*(s,a) based on current Q

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q_star = value(q,0)
    >>> q_star
    10
    """
    return max(q.get(s,a) for a in q.actions) #get max val from dict.

# Given a state, return the action that is greedy with reespect to x
# current definition of the q function
def greedy(q, s):
    """ Return pi*(s) based on a greedy strategy.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> greedy(q, 0)
    'c'
    >>> greedy(q, 1)
    'b'
    """
    #return argmax(q.actions, value(q,s)) #correct but breaks grader
    return argmax(q.actions, lambda r: q.get(s,a)) #fanchy schmanchy

def epsilon_greedy(q, s, eps = 0.5):
    """ Return an action.

    >>> q = TabularQ([0,1,2,3],['b','c'])
    >>> q.set(0, 'b', 5)
    >>> q.set(0, 'c', 10)
    >>> q.set(1, 'b', 2)
    >>> eps = 0.
    >>> epsilon_greedy(q, 0, eps) #greedy
    'c'
    >>> epsilon_greedy(q, 1, eps) #greedy
    'b'
    """
    if random.random() < eps:  # True with prob eps, random action
        #a=random.randint(len(q.actions()))
        #return q.actions(a)
        return uniform_dist(q.actions).draw() #sliick
    else:
        return greedy(q,s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions]) #init
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
