# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for state in self.mdp.getStates():
            self.values[state] = 0.0
        for i in range(self.iterations):
            new_value = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    new_value[state]=0
                    continue
                action_dic={}
                for action in self.mdp.getPossibleActions(state):
                    action_dic[action]=self.getQValue(state, action)
                new_value[state]=max(action_dic.values())
            self.values = new_value.copy()
        

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q=0.0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q += prob*(self.mdp.getReward(state, action, nextState)+self.discount*self.values[nextState])
        return q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        action_dic={}
        if self.mdp.isTerminal(state):
            return None
        for action in self.mdp.getPossibleActions(state):
            action_dic[action]=self.getQValue(state, action)
        return max(action_dic,key=action_dic.get)

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_state = self.mdp.getStates()
        for state in all_state:
            self.values[state] = 0.0
        for i in range(self.iterations):
            state = all_state[i % len(all_state)]
            if self.mdp.isTerminal(state):
                continue
            action_dic={}
            for action in self.mdp.getPossibleActions(state):
                action_dic[action]=self.getQValue(state, action)
            self.values[state]=max(action_dic.values())

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # for state in self.mdp.getStates():
        #     self.values[state] = 0.0
        all_state = self.mdp.getStates()
        priority_queue=util.PriorityQueue()
        predecessor ={}
        ## get all the predecessor
        for state in all_state:
            if self.mdp.isTerminal(state):
                continue
            action_dic={}
            for action in self.mdp.getPossibleActions(state):
                for next_state,_ in self.mdp.getTransitionStatesAndProbs(state, action):
                    if next_state not in predecessor:
                        predecessor[next_state] = set()
                    predecessor[next_state].add(state)
                action_dic[action]=self.getQValue(state, action)
            diff=abs(self.values[state] - max(action_dic.values()))
            priority_queue.update(state,-diff)
        #print(predecessor)

        ##start iterations
        for i in range(self.iterations):
            if priority_queue.isEmpty():
                break
            state = priority_queue.pop()
            if not self.mdp.isTerminal(state):
                action_dic={}
                for action in self.mdp.getPossibleActions(state):
                    action_dic[action]=self.getQValue(state, action)
                self.values[state] = max(action_dic.values())
            for p in predecessor[state]:
                if self.mdp.isTerminal(p):
                    continue
                action_dic={}
                for action in self.mdp.getPossibleActions(p):
                    action_dic[action]=self.getQValue(p, action)
                diff=abs(self.values[p] - max(action_dic.values()))
                if diff > self.theta:
                    priority_queue.update(p,-diff)

