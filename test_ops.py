import numpy as np
import tensorflow as tf


rng = np.random.RandomState(1234)

proba = np.array([[0.2, 0.3, 0.4],
                  [0.4, 0.6, 0.7],
                  [0.3, 0.2, 0.8]])

proba_var = tf.placeholder('float32', name='proba', shape=[None, proba.shape[1]])

def mu():
    """
    \mu = [d_0, d_0, d_0, d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
    \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
    \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]

    Tree indexing
         0
       1   2
      3 4 5 6
    """
    # n_nodes = 2 ** depth + 1
    batch_size = tf.shape(proba_var)[0]
    n_nodes = proba_var.get_shape().as_list()[-1]
    n_leaves = n_nodes + 1
    depth = np.float32(np.log2(n_leaves))

    # decision probabilities.
    # The first n_batch * n_nodes values are d(i)
    # The second n_batch * n_nodes values are 1-d(i)
    decision_p = tf.pack([proba_var, 1 - proba_var])
    flat_decision_p = tf.reshape(decision_p, [-1])

    # zeroth index of each routing probability in the mini-batch
    batch_0_indices = tf.tile(
        tf.expand_dims(tf.range(0, batch_size * n_nodes, n_nodes), 1),
        [1, n_leaves])

    batch_complement_row = tf.concat(1,
        [tf.zeros([1, n_leaves/2]),
         tf.fill([1, n_leaves/2], tf.cast(n_nodes * batch_size, tf.float32))]
    )
    batch_complement_indices =  tf.cast(tf.tile(batch_complement_row, tf.pack([batch_size, 1])), tf.int32)

    # First row of mu
    mu = tf.gather(flat_decision_p, tf.add(batch_0_indices, batch_complement_indices))

    for d in xrange(1, int(depth)):
        indices = tf.range(2 ** d, 2 ** (d + 1)) - 1 # [1, 2]
        tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                          [1, int(2 ** (depth -1 - d + 1))]), [1, -1])
        batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, tf.pack([batch_size, 1])))

        # [1 1 2 2] + [0, 9, 0, 9]
        batch_complement_row = tf.tile(
            tf.concat(1,
                [tf.zeros([1, n_leaves/4]),
                 tf.fill([1, n_leaves/4], tf.cast(n_nodes * batch_size, tf.float32))]
            ),
            [1, n_leaves/2]
        )
        batch_complement_indices =  tf.cast(tf.tile(batch_complement_row, tf.pack([batch_size, 1])), tf.int32)
        mu = tf.mul(mu, tf.gather(flat_decision_p, tf.add(batch_indices, batch_complement_indices)))

        return mu

sess = tf.Session()
data = mu()

sess.run(tf.initialize_all_variables())
result = sess.run(data, feed_dict={proba_var: proba})

print(result)
