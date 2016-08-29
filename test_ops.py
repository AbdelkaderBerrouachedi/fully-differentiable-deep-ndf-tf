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
    batch_size = tf.shape(proba_var)[0]
    n_nodes = proba_var.get_shape().as_list()[-1]
    depth = int(np.log2(n_nodes + 1))  # depth does not include final leaves
    n_leaves = 2 ** depth
    decision_p = tf.pack([proba_var, 1 - proba_var])


    flattened_decision_p = tf.reshape(decision_p, [-1])

    batch_0_indices = tf.tile(tf.expand_dims(tf.range(0, batch_size * n_leaves, n_leaves), 1), [1, n_leaves])
    # [0, 0, 0, 0], [4, 4, 4, 4], [8, 8, 8, 8]

    #batch_complement_indices = tf.tile(tf.expand_dims(tf.concat(0, [tf.zeros([n_leaves / 2]), tf.ones([n_leaves / 2]) *  n_leaves]), 1), [1, n_leaves])
    #batch_complement_indices = tf.tile(tf.expand_dims(tf.range(n_nodes, batch_size * 2 * n_nodes,  2 * n_nodes), 1), [1, n_leaves / 2])
    batch_complement_indices =  tf.tile(tf.concat(0, [tf.zeros([n_leaves/2]), tf.fill([n_leaves/2], tf.cast(batch_size, tf.float32))]), [1, n_leaves])


    #mu = tf.gather(flattened_decision_p, tf.concat(1, [batch_0_indices, batch_complement_indices]))
    return batch_complement_indices
    #for d in xrange(1, int(depth)):
    #    indices = tf.range(2 ** d, 2 ** (d + 1)) - 1 # [1, 2]
    #    tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1), [1, 2]), [1, -1])
    #    batch_indices = tf.tile(tile_indices, tf.pack([batch_size, 1]))
    #    return tile_indices
    #    return tf.gather(flattened_decision_p, batch_indices)


sess = tf.Session()
data = mu()

sess.run(tf.initialize_all_variables())
result = sess.run(data, feed_dict={proba_var: proba})

print(result)
