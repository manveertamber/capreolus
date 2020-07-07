import hashlib
import os
from collections import defaultdict
from copy import copy

import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from profane import ConfigOption
from tqdm import tqdm

from capreolus.searcher import Searcher
from capreolus import evaluator
from capreolus.trainer import Trainer
from capreolus.trainer.tensorflow import TensorFlowTrainer
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


@Trainer.register
class TPUTrainer(TensorFlowTrainer):
    """
    TODO: Contains code specific to TFBERTMaxP (eg: uses two optimizers)
    Need work before this can be used for all rerankers
    """

    module_name = "tputrainer"
    config_spec = TensorFlowTrainer.config_spec + [ConfigOption("decaystep", 3), ConfigOption("decay", 0.96),
                                                   ConfigOption("decaytype", "exponential"), ConfigOption("epochs", 3)]

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.config["lr"])

    def change_lr(self, epoch):
        """
        Apply warm up or decay depending on the current epoch
        """

        warmup_steps = self.config["warmupsteps"]
        if epoch <= warmup_steps:
            return min(self.config["bertlr"] * ((epoch + 1) / warmup_steps), self.config["lr"])
        else:
            # Exponential decay
            if self.config["decaytype"] == "exponential":
                return self.config["bertlr"] * self.config["decay"] ** (
                            (epoch - warmup_steps) / self.config["decaystep"])
            elif self.config["decaytype"] == "linear":
                return self.config["bertlr"] * (1 / (1 + self.config["decay"] * epoch))

    def convert_to_tf_dev_record(self, reranker, dataset):
        """
        Similar to self.convert_to_tf_train_record(), but won't result in multiple files
        """
        dir_name = self.get_tf_record_cache_path(dataset)
        tf_features = []
        tf_record_filenames = []

        for sample in dataset:
            tf_features.extend(reranker.extractor.create_tf_dev_feature(sample))
            if len(tf_features) > 20000:
                tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))
                tf_features = []

        # TPU's require drop_remainder = True. But we cannot drop things from validation dataset
        # As a workaroud, we pad the dataset with the last sample until it reaches the batch size.
        if len(tf_features) % self.config["batch"]:
            num_elements_to_add = self.config["batch"] - (len(tf_features) % self.config["batch"])
            logger.debug("Number of elements to add in the last batch: {}".format(num_elements_to_add))
            element_to_copy = tf_features[-1]
            for i in range(num_elements_to_add):
                tf_features.append(copy(element_to_copy))

        if len(tf_features):
            tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))

        return tf_record_filenames

    def convert_to_tf_train_record(self, reranker, dataset):
        """
        Tensorflow works better if the input data is fed in as tfrecords
        Takes in a dataset,  iterates through it, and creates multiple tf records from it.
        The exact structure of the tfrecords is defined by reranker.extractor. For example, see EmbedText.get_tf_feature()
        """
        dir_name = self.get_tf_record_cache_path(dataset)

        total_samples = dataset.get_total_samples()
        tf_features = []
        tf_record_filenames = []

        for niter in tqdm(range(0, self.config["niters"]), desc="Converting data to tf records"):
            for sample_idx, sample in enumerate(dataset):
                tf_features.extend(reranker.extractor.create_tf_train_feature(sample))

                if len(tf_features) > 20000:
                    tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))
                    tf_features = []

                if sample_idx + 1 >= self.config["itersize"] * self.config["batch"]:
                    break

        if len(tf_features):
            tf_record_filenames.append(self.write_tf_record_to_file(dir_name, tf_features))

        return tf_record_filenames

    def load_tf_train_records_from_file(self, reranker, filenames, batch_size):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        tf_records_dataset = raw_dataset.batch(batch_size, drop_remainder=True).map(
            reranker.extractor.parse_tf_train_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return tf_records_dataset

    def load_cached_tf_train_records(self, reranker, dataset, batch_size):
        logger.info("Loading TF records from cache")
        cache_dir = self.get_tf_record_cache_path(dataset)
        filenames = tf.io.gfile.listdir(cache_dir)
        filenames = ["{0}/{1}".format(cache_dir, name) for name in filenames]

        return self.load_tf_train_records_from_file(reranker, filenames, batch_size)

    def load_tf_dev_records_from_file(self, reranker, filenames, batch_size):
        raw_dataset = tf.data.TFRecordDataset(filenames)
        tf_records_dataset = raw_dataset.batch(batch_size, drop_remainder=True).map(
            reranker.extractor.parse_tf_dev_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return tf_records_dataset

    def load_cached_tf_dev_records(self, reranker, dataset, batch_size):
        logger.info("Loading TF records from cache")
        cache_dir = self.get_tf_record_cache_path(dataset)
        filenames = tf.io.gfile.listdir(cache_dir)
        filenames = ["{0}/{1}".format(cache_dir, name) for name in filenames]

        return self.load_tf_dev_records_from_file(reranker, filenames, batch_size)

    def get_tf_dev_records(self, reranker, dataset):
        """
        1. Returns tf records from cache (disk) if applicable
        2. Else, converts the dataset into tf records, writes them to disk, and returns them
        """
        if self.config["usecache"] and self.cache_exists(dataset):
            return self.load_cached_tf_dev_records(reranker, dataset, self.config["batch"])
        else:
            tf_record_filenames = self.convert_to_tf_dev_record(reranker, dataset)
            # TODO use actual batch size here. see issue #52
            return self.load_tf_dev_records_from_file(reranker, tf_record_filenames, self.config["batch"])

    def get_tf_train_records(self, reranker, dataset):
        """
        1. Returns tf records from cache (disk) if applicable
        2. Else, converts the dataset into tf records, writes them to disk, and returns them
        """

        if self.config["usecache"] and self.cache_exists(dataset):
            return self.load_cached_tf_train_records(reranker, dataset, self.config["batch"])
        else:
            tf_record_filenames = self.convert_to_tf_train_record(reranker, dataset)
            return self.load_tf_train_records_from_file(reranker, tf_record_filenames, self.config["batch"])

    def train(self, reranker, train_dataset, train_output_path, dev_data, dev_output_path, qrels, metric,
              relevance_level=1):
        if self.tpu:
            train_output_path = "{0}/{1}/{2}".format(
                self.config["storage"], "train_output", hashlib.md5(str(train_output_path).encode("utf-8")).hexdigest()
            )

        os.makedirs(dev_output_path, exist_ok=True)

        train_records = self.get_tf_train_records(reranker, train_dataset)
        dev_records = self.get_tf_dev_records(reranker, dev_data)
        dev_dist_dataset = self.strategy.experimental_distribute_dataset(dev_records)

        strategy_scope = self.strategy.scope()
        with strategy_scope:
            reranker.build_model()
            wrapped_model = self.get_model(reranker.model)
            loss_object = self.get_loss(self.config["loss"])
            optimizer_1 = self.get_optimizer()
            optimizer_2 = tf.keras.optimizers.Adam(learning_rate=self.config["bertlr"])

            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config["batch"])

        def train_step(inputs):
            data, labels = inputs

            with tf.GradientTape() as tape:
                predictions = wrapped_model(data, training=True)
                loss = compute_loss(labels, predictions)

            gradients = tape.gradient(loss, wrapped_model.trainable_variables)
            bert_variables = [(gradients[i], variable) for i, variable in enumerate(wrapped_model.trainable_variables)
                              if 'bert' in variable.name]
            classifier_vars = [(gradients[i], variable) for i, variable in enumerate(wrapped_model.trainable_variables)
                               if 'classifier' in variable.name]
            other_vars = [(gradients[i], variable) for i, variable in enumerate(wrapped_model.trainable_variables) if
                          'bert' not in variable.name and 'classifier' not in variable.name]
            # Making sure that we did not miss any variables
            assert len(other_vars) == 0

            optimizer_1.apply_gradients(classifier_vars)
            optimizer_2.apply_gradients(bert_variables)

            return loss

        def test_step(inputs):
            data, labels = inputs
            predictions = wrapped_model.predict_step(data)

            return predictions

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = self.strategy.run(train_step, args=(dataset_inputs,))

            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return self.strategy.run(test_step, args=(dataset_inputs,))

        best_metric = -np.inf
        epoch = 0
        num_batches = 0
        total_loss = 0
        iter_bar = tqdm(total=self.config["itersize"])

        initial_lr = self.change_lr(epoch)
        K.set_value(optimizer_2.lr, K.get_value(initial_lr))
        train_records = train_records.shuffle(100000).repeat(count=3)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_records)

        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            train_loss = total_loss / num_batches
            num_batches += 1
            iter_bar.update(1)

            if num_batches % self.config["itersize"] == 0:
                epoch += 1
                # if epoch > self.config["niters"]:
                #     break

                # Do warmup and decay
                new_lr = self.change_lr(epoch)
                K.set_value(optimizer_2.lr, K.get_value(new_lr))

                iter_bar.close()
                iter_bar = tqdm(total=self.config["itersize"])
                logger.info("train_loss for epoch {} is {}".format(epoch, train_loss))
                train_loss = 0
                total_loss = 0

                if epoch % self.config["validatefreq"] == 0:
                    predictions = []
                    for x in tqdm(dev_dist_dataset, desc="validation"):
                        pred_batch = distributed_test_step(x).values if self.strategy.num_replicas_in_sync > 1 else [
                            distributed_test_step(x)]
                        for p in pred_batch:
                            predictions.extend(p)

                    trec_preds = self.get_preds_in_trec_format(predictions, dev_data)
                    metrics = evaluator.eval_runs(trec_preds, dict(qrels), evaluator.DEFAULT_METRICS, relevance_level)
                    logger.info("dev metrics: %s",
                                " ".join([f"{metric}={v:0.3f}" for metric, v in sorted(metrics.items())]))
                    if metrics[metric] > best_metric:
                        logger.info("Writing checkpoint")
                        best_metric = metrics[metric]
                        wrapped_model.save_weights("{0}/dev.best".format(train_output_path))

    @staticmethod
    def get_preds_in_trec_format(predictions, dev_data):
        """
        Takes in a list of predictions and returns a dict that can be fed into pytrec_eval
        As a side effect, also writes the predictions into a file in the trec format
        """
        logger.debug("There are {} predictions".format(len(predictions)))
        pred_dict = defaultdict(lambda: dict())

        for i, (qid, docid) in enumerate(dev_data.get_qid_docid_pairs()):
            # Pytrec_eval has problems with high precision floats
            pred_dict[qid][docid] = predictions[i].numpy().astype(np.float16).item()

        return dict(pred_dict)

    def predict(self, reranker, pred_data, pred_fn):
        pred_records = self.get_tf_dev_records(reranker, pred_data)
        pred_dist_dataset = self.strategy.experimental_distribute_dataset(pred_records)

        strategy_scope = self.strategy.scope()

        with strategy_scope:
            wrapped_model = self.get_model(reranker.model)

        def test_step(inputs):
            data, labels = inputs
            predictions = wrapped_model.predict_step(data)

            return predictions

        @tf.function
        def distributed_test_step(dataset_inputs):
            return self.strategy.run(test_step, args=(dataset_inputs,))

        predictions = []
        for x in pred_dist_dataset:
            pred_batch = distributed_test_step(x).values
            for p in pred_batch:
                predictions.extend(p)

        trec_preds = self.get_preds_in_trec_format(predictions, pred_data)
        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(trec_preds, pred_fn)

        return trec_preds
