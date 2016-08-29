import numpy as np
import skflow
import tensorflow as tf


def neural_decision_tree(X, y
                         max_depth=3,
                         random_state=42,
                         trainable=True,
                         name='neural_decision_tree'):
    """A Single shallow neural decision tree."""
    with tf.variable_op_scope([X, y], name, 'neural_decision_tree'):
        n_hidden_units = 2 ** (max_depth - 1) - 1
        n_classes = y.get_shape().as_list()[-1]

        fully_connected = skflow.ops.deep_network(
                X,
                n_layers=1,
                hidden_units=n_hidden_units,
                activation='relu',
                keep_prob=0.5,
                batch_norm=False,
                random_state=random_state)

        weights_shape = [n_leaf, n_classes]
        weights_init = skflow.initializers.get('glorot_normal')(
                weights_shape, seed=random_state)
        W_leaf = skflow.tensor.variable('weight_leaf',
                                        shape=weights_shape,
                                        initializer=,
                                        trainable=trainable)
        decision_p = tf.nn.sigmoid(fully_connected)
        leaf_p = tf.nn.softmax(W_leaf)

        return decision_p, leaf_p


def routing_probability(decision_p, max_depth, name=None):
    """routing_probability.

    Creates a routing probability matrix \mu of size [n_leaves].
    First, \mu is initialized using the root node d, 1 - d.
    To efficiently implement this routing, a giant vector (matrix)
    that contains all d and 1 - d from all decision nodes is created.

    For a depth = 2 tree, the routing probability for each leaf node
    can be easily computed by multiplying the following vectors elementwise.

    \mu = [d_0, d_0, d_0, d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
    \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
    \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]

    Tree indexing
         0
       1   2
      3 4 5 6

    Parameters:
    -----------

    decision_p : tf.Tensor of shape [N_BATCH, N_NODES]
    """
    with tf.variable_op_scope([decision_p], name, 'routing_probability'):
        # compute the complement of d, which is 1-d
        # where d is the sigmoid of the fully-connected output
        decision_p_comp = tf.sub(tf.ones_like(decision_p), decision_p)

        # concatenate both d and 1-d [N_BATCHES, 2 * N_NODES]
        flat_decision_p = tf.concat(1, [decision_p, decision_p_comp])

        tf.gather(flat_decision_p, [None, 0, N_NODES])

        # first layer is just the first node probabilitya
        for d in xrange(max_depth):
            indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
