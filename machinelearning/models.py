import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return(nn.DotProduct(x,self.get_weights()))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        pred=nn.as_scalar(self.run(x))
        return(1 if pred>=0 else -1)
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            batch_size=1
            correct=True
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x,nn.as_scalar(y))
                    correct=False
            if correct:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.d1=50
        self.w1=nn.Parameter(1,self.d1)
        self.b1=nn.Parameter(1,self.d1)
        self.w2=nn.Parameter(self.d1,1)
        self.b2=nn.Parameter(1,1)
        self.batch_size=20
        self.alpha=-0.05
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        l1=nn.Linear(x,self.w1)
        l1_y=nn.AddBias(l1,self.b1)
        l1_relu=nn.ReLU(l1_y)
        l2=nn.Linear(l1_relu,self.w2)
        l2_y=nn.AddBias(l2,self.b2)
        return(l2_y)
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return(nn.SquareLoss(self.run(x), y))
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss=self.get_loss(x,y)
                all_p=[self.w1,self.b1,self.w2,self.b2]
                grad=nn.gradients(loss,all_p)
                for p,p_grad in zip(all_p,grad):
                    p.update(p_grad,self.alpha)
            if  nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) <=0.01:
                break
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.d1=200
        self.w1=nn.Parameter(784,self.d1)
        self.b1=nn.Parameter(1,self.d1)
        self.w2=nn.Parameter(self.d1,10)
        self.b2=nn.Parameter(1,10)
        self.batch_size=100
        self.alpha=-0.05
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        l1=nn.Linear(x,self.w1)
        l1_y=nn.AddBias(l1,self.b1)
        l1_relu=nn.ReLU(l1_y)
        l2=nn.Linear(l1_relu,self.w2)
        l2_y=nn.AddBias(l2,self.b2)
        return(l2_y)
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return(nn.SoftmaxLoss(self.run(x),y))

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss=self.get_loss(x,y)
                all_p=[self.w1,self.b1,self.w2,self.b2]
                grad=nn.gradients(loss,all_p)
                for p,p_grad in zip(all_p,grad):
                    p.update(p_grad,self.alpha)
            if  dataset.get_validation_accuracy() >= 0.976:
                #print(dataset.get_validation_accuracy())
                break
class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = -0.1
        self.numTrainingGames = 11000
        self.batch_size = 32
        self.d1=256
        self.d2=128
        self.w1=nn.Parameter(state_dim,self.d1)
        self.b1=nn.Parameter(1,self.d1)
        self.w2=nn.Parameter(self.d1,self.d2)
        self.b2=nn.Parameter(1,self.d2)
        self.w3=nn.Parameter(self.d2,action_dim)
        self.b3=nn.Parameter(1,action_dim)
        self.parameters =[self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return(nn.SquareLoss(self.run(states),Q_target))
    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"
        l1=nn.Linear(states,self.w1)
        l1_y=nn.AddBias(l1,self.b1)
        l1_relu=nn.ReLU(l1_y)
        l2=nn.Linear(l1_relu,self.w2)
        l2_y=nn.AddBias(l2,self.b2)
        l2_relu=nn.ReLU(l2_y)
        l3=nn.Linear(l2_relu,self.w3)
        l3_y=nn.AddBias(l3,self.b3)
        return(l3_y)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states,Q_target)
        grad=nn.gradients(loss,self.parameters)
        for p,p_grad in zip(self.parameters,grad):
                p.update(p_grad,self.learning_rate)