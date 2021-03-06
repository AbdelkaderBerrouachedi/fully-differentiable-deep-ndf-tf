import numpy as np
import skflow
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

import ndf

DEPTH = 4  # Depth of a tree (this includes the leaf probabilities)
N_LEAF = 2 ** (DEPTH - 1)  # Number of leaf nodes
N_DECISION_NODES = 2 ** (DEPTH - 1) - 1  # These are all nodes but the leaves
N_BATCH = 3
N_LABELS = 10


class TestNDF(tf.test.TestCase):
    def test_routing_probability(self):
        rng = np.random.RandomState(1234)

        decision_p = rng.beta(
                2, 2, N_DECISION_NODES * N_BATCH).reshape(
                        N_BATCH, N_DECISION_NODES)

        decision_p = tf.constant(decision_p)
        with self.test_session():
            routing_proba = ndf.routing_probability(decision_p).eval()

            self.assertEquals(routing_proba.shape, (N_BATCH, N_LEAF))
            self.assertTrue(np.all(routing_proba < 1))
            self.assertTrue(np.all(routing_proba > 0))

    def test_leaf_probabilities(self):
        rng = np.random.RandomState(1234)

        decision_p = rng.beta(
                2, 2, N_DECISION_NODES * N_BATCH).reshape(
                        N_BATCH, N_DECISION_NODES).astype(np.float32)
        decision_p = tf.constant(decision_p)

        w_l = rng.uniform(-2, 2, [N_LEAF, N_LABELS]).astype(np.float32)
        w_l = tf.constant(w_l)

        with self.test_session():
            leaf_p = tf.nn.softmax(w_l)
            routing_proba = ndf.routing_probability(decision_p)
            py_x_tree = ndf.leaf_probability(routing_proba, leaf_p).eval()

            self.assertEqual(py_x_tree.shape, (N_BATCH, N_LABELS))
            self.assertTrue(np.all(routing_proba < 1))
            self.assertTrue(np.all(routing_proba > 0))


    def test_model_op(self):
        rng = np.random.RandomState(1234)
        X = tf.placeholder('float32', name='X', shape=[None, 5])
        y = tf.placeholder('float32', name='y', shape=[None, 10])
        model = ndf.neural_decision_tree(X, y)

        X_data = rng.randn(10, 5)
        y_data = np.eye(10)[rng.choice(np.arange(10), 10)]
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            preds, loss = sess.run(model, feed_dict={X: X_data, y: y_data})

    def test_skflow_tree(self):
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowEstimator(
                ndf.neural_decision_tree_classifier,
                n_classes=3,
                optimizer='Adagrad',
                learning_rate=0.1,
                batch_size=100,
                verbose=True)
        classifier.fit(iris.data, iris.target, logdir='./model')
        score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
        print(score)

    def test_skflow_forest(self):
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowEstimator(
                ndf.neural_decision_forest_classifier,
                n_classes=3,
                optimizer='Adagrad',
                learning_rate=0.1,
                verbose=True)
        classifier.fit(iris.data, iris.target, logdir='./model')
        score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
        print(score)
        print(classifier.predict_proba(iris.data[:5, :]))

    def test_dt(self):
        iris = datasets.load_iris()
        classifier = DecisionTreeClassifier(max_depth=4)
        classifier.fit(iris.data, iris.target)
        score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
        print(score)

    def test_flavour(self):
        import pandas as pd
        np.random.seed(123)
        df = pd.read_csv('./data/flavour_noleak.csv')
        indices = np.arange(df.shape[0])
        np.random.shuffle(indices)
        y = df.pop('signal').values[indices[:1000]]
        print(y)

        X = df.values[indices[:10000]]

        classifier = skflow.TensorFlowEstimator(
                ndf.neural_decision_forest_classifier,
                n_classes=2,
                optimizer='Adagrad',
                learning_rate=0.1,
                verbose=True)
        classifier.fit(X, y, logdir='./model')
        score = metrics.accuracy_score(y, classifier.predict(X))
        print(score)
