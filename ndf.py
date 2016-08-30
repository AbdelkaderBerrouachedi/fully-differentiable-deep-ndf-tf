import numpy as np
import skflow
import tensorflow as tf


def get_tensor_shape(tensor):
    return tensor.get_shape().as_list()


def flatten_tensor(tensor):
    return tf.reshape(tensor, [-1])


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def categorical_crossentropy(output, target, name=None):
    with tf.variable_op_scope([output, target], name, 'categorical_xentropy'):
        output /= tf.reduce_sum(output,
                                reduction_indices=len(output.get_shape()) - 1,
                                keep_dims=True)
        epsilon = _to_tensor(1e-7, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        return - tf.reduce_sum(target * tf.log(output),
                               reduction_indices=len(output.get_shape()) - 1)

def routing_probability(decision_p, name=None):
    """routing_probability.

    Creates a routing probability matrix \mu of size [n_leaves].
    First, \mu is initialized using the root node d, 1 - d.
    To efficiently implement this routing, a giant vector (matrix)
    that contains all d and 1 - d from all decision nodes is created.

    For a depth = 4 tree, the routing probability for each leaf node
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

    Returns:
    --------
    routing_proba : tf.Tensor of shape [N_BATCH, N_LEAVES]
        The routing probability of landing in a given leaf node.

    """
    with tf.variable_op_scope([decision_p], name, 'routing_probability'):
        # determine various tensor shapes
        batch_size = tf.shape(decision_p)[0]
        n_decision_nodes = get_tensor_shape(decision_p)[-1]
        n_leaves = n_decision_nodes + 1
        tree_depth = np.int32(np.log2(n_leaves)) + 1

        # Decision Probabilities
        # The first n_batch * n_decision_nodes values are d(i)
        # The second n_batch * n_decision_nodes values are 1 - d(i)
        flat_decision_p = flatten_tensor(tf.pack([decision_p, 1 - decision_p]))

        # zeroth index of each routing probability in the mini-batch
        batch_0_indices = tf.tile(
            tf.expand_dims(
                tf.range(0, batch_size*n_decision_nodes, n_decision_nodes), 1),
            [1, n_leaves])

        # Let batch_size * n_leaves = N_D. flat_decision_p[N_D] will return
        # 1 - d of the first root node in the first tree.
        batch_complement_row = tf.concat(1,
            [tf.zeros([1, n_leaves/2]),
             tf.fill([1, n_leaves/2],
                 tf.to_float(n_decision_nodes * batch_size))]
        )
        batch_complement_indices = tf.to_int32(
                tf.tile(batch_complement_row, tf.pack([batch_size, 1])))

        # Finally the first row of the routing probability matrix computation
        routing_proba = tf.gather(
                flat_decision_p,
                tf.add(batch_0_indices, batch_complement_indices))

        # from the second layer to the last layer, we make the decision nodes
        for d in xrange(2, tree_depth):
            indices = tf.range(2 ** (d - 1), 2 ** d) - 1 # [2, 4]
            tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                              [1, 2 ** (tree_depth - d)]),
                                      [1, -1])
            batch_indices = tf.add(
                    batch_0_indices,
                    tf.tile(tile_indices, tf.pack([batch_size, 1])))

            batch_complement_row = tf.tile(
                tf.concat(1,
                    [tf.zeros([1, n_leaves/(2**d)]),
                     tf.fill([1, n_leaves/(2**d)],
                         tf.to_float(n_decision_nodes * batch_size))]
                ),
                [1, 2 ** (d - 1)]
            )

            batch_complement_indices = tf.to_int32(
                    tf.tile(batch_complement_row, tf.pack([batch_size, 1])))

            routing_update = tf.gather(
                    flat_decision_p,
                    tf.add(batch_indices, batch_complement_indices))
            routing_proba = tf.mul(routing_proba, routing_update)

        return routing_proba


def leaf_probability(routing_proba, leaf_p, name=None):
    with tf.variable_op_scope([routing_proba, leaf_p], name, 'leaf_proba'):
        batch_size = tf.shape(routing_proba)[0]
        n_labels = get_tensor_shape(leaf_p)[-1]

        py_x_tree = tf.reduce_mean(
                tf.mul(
                    tf.tile(tf.expand_dims(routing_proba, 2),
                        [1, 1, n_labels]),
                    tf.tile(tf.expand_dims(leaf_p, 0),
                        tf.pack([batch_size, 1, 1]))
                ),
            1)

        return py_x_tree


def neural_decision_tree(X, y,
                         max_depth=3,
                         random_state=42,
                         trainable=True,
                         name='neural_decision_tree'):
    """A Single shallow neural decision tree."""
    with tf.variable_op_scope([X, y], name, 'neural_decision_tree'):
        n_leaf = 2 ** (max_depth - 1)
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
        weights_init = tf.random_uniform_initializer(-2, 2, seed=random_state)
        W_leaf = skflow.tensor.variable('weight_leaf',
                                        shape=weights_shape,
                                        initializer=weights_init,
                                        trainable=trainable)

        # apply tree layer
        decision_p = tf.nn.sigmoid(fully_connected)
        leaf_p = tf.nn.softmax(W_leaf)
        routing_proba = routing_probability(decision_p)

        predictions = leaf_probability(routing_proba, leaf_p)
        loss = tf.reduce_sum(categorical_crossentropy(predictions, y))

        return predictions, loss
