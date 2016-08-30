import numpy as np
import tensorflow as tf

DEPTH = 4  # Depth of a tree (this includes the leaf probabilities)
N_LEAF = 2 ** (DEPTH - 1)  # Number of leaf nodes
N_DECISION_NODES = 2 ** (DEPTH - 1) - 1  # These are all nodes but the leaves
N_BATCH = 3
N_LABELS = 10

rng = np.random.RandomState(1234)

proba = rng.beta(2, 2, N_DECISION_NODES * N_BATCH).reshape(
            N_BATCH, N_DECISION_NODES)

print(proba)

proba_var = tf.placeholder('float32', name='proba', shape=[None, proba.shape[1]])

def mu_calc():
    """
    \mu = [d_0, d_0, d_0, d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
    \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
    \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]

    Tree indexing
         0
       1   2
      3 4 5 6
    """
    batch_size = tf.shape(proba_var)[0]
    n_decision_nodes = proba_var.get_shape().as_list()[-1]
    n_leaves = n_decision_nodes + 1
    tree_depth = np.int64(np.log2(n_leaves) + 1)

    # decision probabilities.
    # The first n_batch * n_decision_nodes values are d(i)
    # The second n_batch * n_decision_nodes values are 1-d(i)
    decision_p = tf.pack([proba_var, 1 - proba_var])
    flat_decision_p = tf.reshape(decision_p, [-1])

    # zeroth index of each routing probability in the mini-batch
    batch_0_indices = tf.tile(
        tf.expand_dims(tf.range(0, batch_size * n_decision_nodes, n_decision_nodes), 1),
        [1, n_leaves])

    batch_complement_row = tf.concat(1,
        [tf.zeros([1, n_leaves/2]),
         tf.fill([1, n_leaves/2], tf.to_float(n_decision_nodes * batch_size))]
    )
    batch_complement_indices = tf.to_int32(tf.tile(batch_complement_row, tf.pack([batch_size, 1])))

    # First row of mu
    mu = tf.gather(flat_decision_p, tf.add(batch_0_indices, batch_complement_indices))

    for d in xrange(2, tree_depth):
        indices = tf.range(2 ** (d - 1), 2 ** d) - 1 # [2, 4]
        tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                          [1, 2 ** (tree_depth - d)]), [1, -1])
        batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, tf.pack([batch_size, 1])))

        batch_complement_row = tf.tile(
            tf.concat(1,
                [tf.zeros([1, n_leaves/(2**d)]),
                 tf.fill([1, n_leaves/(2**d)], tf.to_float(n_decision_nodes * batch_size))]
            ),
            [1, 2 ** (d - 1)]
        )

        batch_complement_indices = tf.to_int32(tf.tile(batch_complement_row, tf.pack([batch_size, 1])))
        mu = tf.mul(mu, tf.gather(flat_decision_p, tf.add(batch_indices, batch_complement_indices)))
    return mu


def pyx(mu):
    batch_size = tf.shape(mu)[0]
    w_l = tf.Variable(tf.random_uniform([N_LEAF, N_LABELS], -2, 2, seed=1))
    leaf_p = tf.nn.softmax(w_l)
    return tf.reduce_mean(
            tf.mul(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABELS]),
                  tf.tile(tf.expand_dims(leaf_p, 0), tf.pack([batch_size, 1, 1]))), 1)

sess = tf.Session()
data = pyx(mu_calc())

sess.run(tf.initialize_all_variables())
result = sess.run(data, feed_dict={proba_var: proba})

print(result)
