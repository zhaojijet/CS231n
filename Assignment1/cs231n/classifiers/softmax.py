import numpy as np
from random import shuffle

def softmaxprob(score):
    if len(score.shape) > 1:
        score_max = np.max(score, 1, keepdims = True)
    else:
        score_max = np.max(score)
    score_exp = np.exp(score - score_max)
    if len(score.shape) > 1:
        return score_exp / np.sum(score_exp, 1, keepdims = True)
    else:
        return score_exp / np.sum(score_exp)

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
      prob = softmaxprob(np.dot(X[i], W))
      loss += -np.log(prob[y[i]])
      prob[y[i]] -= 1
      dW += np.dot(X[i].reshape(1, -1).T, prob.reshape(1, -1))
  loss /= num_train 
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  prob = softmaxprob(np.dot(X, W))
  loss = np.sum(-np.log(prob[np.arange(num_train), y])) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  grad = prob.copy()
  grad[np.arange(num_train), y] -= 1.
  dW = np.dot(X.T, grad) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

