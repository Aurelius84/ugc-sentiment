"""
Multi-label classification models.

https://en.wikipedia.org/wiki/Multi-label_classification
"""
import sys
import warnings
import pickle
import numpy as np
from math import ceil

from sklearn import linear_model as lm

from keras import backend as K
from keras.models import Model
K.set_learning_phase(1)


class MLC(Model):
    """
    Multi-label classifier.

    Extends keras.models.Graph. Provides additional functionality and metrics
    specific to multi-label classification.
    """

    def _construct_thresholds(self, probs, targets, top_k=None):
        assert probs.shape == targets.shape, \
            "The shape of predictions should match the shape of targets."
        nb_samples, nb_labels = targets.shape
        top_k = top_k or nb_labels

        # Sort predicted probabilities in descending order
        idx = np.argsort(probs, axis=1)[:, :-(top_k + 1):-1]
        p_sorted = np.vstack([probs[i, idx[i]] for i in range(len(idx))])
        t_sorted = np.vstack([targets[i, idx[i]] for i in range(len(idx))])

        # Compute F-1 measures for every possible threshold position
        F1 = []
        TP = np.zeros(nb_samples)
        FN = t_sorted.sum(axis=1)
        FP = np.zeros(nb_samples)
        for i in range(top_k):
            TP += t_sorted[:, i]
            FN -= t_sorted[:, i]
            FP += 1 - t_sorted[:, i]
            F1.append(2 * TP / (2 * TP + FN + FP))
        F1 = np.vstack(F1).T
        # Find the thresholds
        row = np.arange(nb_samples)
        col = F1.argmax(axis=1)
        p_sorted = np.hstack([p_sorted, np.zeros(nb_samples)[:, None]])
        # T = p_sorted[row, col][:,None]
        ratio = 0.382
        T = (ratio * p_sorted[row, col] +
             (1 - ratio) * p_sorted[row, col + 1])[:, None]

        return T

    def fit_thresholds(self,
                       data,
                       alpha,
                       batch_size=128,
                       y_name=['Y0', 'Y1'],
                       verbose=0,
                       validation_data=None,
                       cv=None,
                       top_k=None,
                       input_sparse=False,
                       use_hidden_feature=False,
                       vocab_size=None):

        inputs = np.hstack([data[k] for k in self.input_names])
        if use_hidden_feature:
            # get outputs of conv layer
            vec_model = K.function([self.get_layer('X').input],
                                   [self.get_layer('H').output])
            batch_num = int(ceil(inputs.shape[0] / float(batch_size)))
            for current_batch in range(batch_num):
                start = current_batch * batch_size
                step = batch_size if current_batch + 1 != batch_num else inputs.shape[0] - current_batch * batch_size
                batch_inputs = inputs[start:start + step]
                batch_outputs = vec_model([batch_inputs])[0]
                t_inputs = batch_outputs if current_batch == 0 else np.concatenate(
                    (t_inputs, batch_outputs), axis=0)

        # now only support matrix average and one-hot vecotor representation
        # TODO sparse word index should be transformed to one-hot vecotor
        elif input_sparse and isinstance(vocab_size, int):
            t_inputs = np.zeros((inputs.shape[0], vocab_size))
            for i in range(inputs.shape[0]):
                t_inputs[i][inputs[i]] = 1
        else:
            t_inputs = np.average(
                inputs, axis=1) if len(inputs.shape) == 3 else inputs
        probs = self.predict(data, batch_size=batch_size)

        probs = dict(zip(y_name, probs))
        targets = {k: data[k] for k in self.output_names}

        if isinstance(alpha, list):
            if validation_data is None and cv is None:
                warnings.warn("Neither validation data, nor the number of "
                              "cross-validation folds is provided. "
                              "The alpha parameter for threshold model will "
                              "be selected based on the default "
                              "cross-validation procedure in RidgeCV.")
            elif validation_data is not None:
                val_inputs = np.hstack(
                    [validation_data[k] for k in self.input_names])
                if use_hidden_feature:
                    batch_num = int(val_inputs.shape[0] / float(batch_size))
                    for current_batch in range(batch_num):
                        start = current_batch * batch_size
                        step = batch_size if current_batch + 1 != batch_num else val_inputs.shape[0] - current_batch * batch_size
                        batch_inputs = val_inputs[start:start + step]
                        batch_outputs = vec_model([batch_inputs])[0]
                        val_t_inputs = batch_outputs if current_batch == 0 else np.concatenate(
                            (val_t_inputs, batch_outputs), axis=0)
                elif input_sparse and isinstance(vocab_size, int):
                    val_t_inputs = np.zeros((val_inputs.shape[0], vocab_size))
                    for i in range(val_inputs.shape[0]):
                        val_t_inputs[i][val_inputs[i]] = 1
                else:
                    val_t_inputs = np.average(
                        val_inputs,
                        axis=1) if len(val_inputs.shape) == 3 else val_inputs
                # val_t_inputs = vec_model([val_inputs])[0]
                val_probs = self.predict(val_inputs)
                val_probs = dict(zip(y_name, val_probs))
                val_targets = {
                    k: validation_data[k]
                    for k in self.output_names
                }

        if verbose:
            sys.stdout.write("Constructing thresholds.")
            sys.stdout.flush()

        t_models = {}
        for k in self.output_names:
            if verbose:
                sys.stdout.write(".")
                sys.stdout.flush()

            T = self._construct_thresholds(probs[k], targets[k])
            # init alpha
            alpha_best = 0
            if isinstance(alpha, list):
                if validation_data is not None:
                    val_T = self._construct_thresholds(
                        val_probs[k], val_targets[k], top_k=top_k)
                    score_best, alpha_best = -np.Inf, None
                    for ind, a in enumerate(alpha):
                        print('fit threshold iter number: %d' % ind)
                        model = lm.Ridge(alpha=a).fit(t_inputs, T)
                        score = model.score(val_t_inputs.astype(float), val_T)
                        if score > score_best:
                            score_best, alpha_best = score, a
                else:
                    model = lm.RidgeCV(alphas=alpha, cv=cv).fit(t_inputs, T)
                    alpha_best = model.alpha_

            t_models[k] = lm.Ridge(alpha=alpha_best)
            t_models[k].fit(t_inputs, T)
        self.t_models = t_models
        pickle.dump(t_models, open('../docs/model/checkpoints/t_models', 'wb'))
        if verbose:
            sys.stdout.write("Done.\n")
            sys.stdout.flush()

    def threshold(self,
                  data,
                  verbose=0,
                  batch_size=128,
                  use_hidden_feature=False):
        inputs = np.hstack([data[k] for k in self.input_names])
        if use_hidden_feature:
            # get outputs of conv layer
            vec_model = K.function([self.get_layer('X').input],
                                   [self.get_layer('H').output])
            batch_num = int(ceil(inputs.shape[0] / float(batch_size)))
            for current_batch in range(batch_num):
                start = current_batch * batch_size
                step = batch_size if current_batch + 1 != batch_num else inputs.shape[0] - current_batch * batch_size
                batch_inputs = inputs[start:start + step]
                batch_outputs = vec_model([batch_inputs])[0]
                t_inputs = batch_outputs if current_batch == 0 else np.concatenate(
                    (t_inputs, batch_outputs), axis=0)
        else:
            t_inputs = np.average(
                inputs, axis=1) if len(inputs.shape) == 3 else inputs

        if verbose:
            sys.stdout.write("Thresholding...\n")
            sys.stdout.flush()

        T = {
            k: self.t_models[k].predict(t_inputs.astype(float))
            for k in self.output_names
        }

        if verbose:
            sys.stdout.write("Done.\n")
            sys.stdout.flush()

        return T

    def predict_threshold(self,
                          data,
                          batch_size=128,
                          y_name=['Y0', 'Y1'],
                          use_hidden_feature=False,
                          verbose=0):
        inputs = np.hstack([data[k] for k in self.input_names])
        probs = self.predict(
            inputs.astype(float), batch_size=batch_size, verbose=verbose)

        probs = dict(zip(y_name, probs))
        T = self.threshold(
            data,
            verbose=verbose,
            batch_size=batch_size,
            use_hidden_feature=use_hidden_feature)
        preds = {k: probs[k] >= T[k] for k in self.output_names}
        return probs, preds

    def predict_combine(self,
                        data,
                        batch_size=128,
                        y_name=['Y0', 'Y1'],
                        use_hidden_feature=False,
                        verbose=0):
        inputs = np.hstack([data[k] for k in self.input_names])
        probs = self.predict(
            inputs.astype(float), batch_size=batch_size, verbose=verbose)

        probs = dict(zip(y_name, probs))
        T = self.threshold(
            data,
            verbose=verbose,
            batch_size=batch_size,
            use_hidden_feature=use_hidden_feature)
        preds_by_thres = {k: probs[k] >= T[k] for k in self.output_names}
        max_ind = {
            k: np.max(probs[k], axis=1).reshape(probs[k].shape[0], 1)
            for k in self.output_names
        }
        preds_amx = {k: probs[k] >= max_ind[k] for k in self.output_names}

        # combine result
        preds_all = {
            k: preds_by_thres[k] + preds_amx[k]
            for k in self.output_names
        }
        return probs, preds_all
