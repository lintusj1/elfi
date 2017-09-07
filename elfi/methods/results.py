"""Containers for results from inference."""

import io
import logging
import sys
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

import elfi.visualization.visualization as vis
from elfi.utils import is_array

logger = logging.getLogger(__name__)


class ParameterInferenceResult:
    """Base class for results."""

    def __init__(self, outputs, parameter_names, n_sim, method_name, **kwargs):
        """Initialize result.

        Parameters
        ----------
        outputs : dict
            Dictionary with outputs from the nodes, e.g. samples.
        parameter_names : list
            Names of the parameter nodes
        method_name : string
            Name of inference method.

        """
        self.outputs = outputs.copy()
        self.parameter_names = parameter_names  # type: list
        self.n_sim = n_sim
        self.method_name = method_name


class OptimizationResult(ParameterInferenceResult):
    """Base class for results from optimization."""

    def __init__(self, x_min, **kwargs):
        """Initialize result.

        Parameters
        ----------
        x_min
            The optimized parameters
        **kwargs
            See `ParameterInferenceResult`

        """
        super(OptimizationResult, self).__init__(**kwargs)
        self.x_min = x_min


class OutputSampleMixin:
    """Add sample specific common properties to sample like classes.

    Notes
    -----
    The following attributes must be present in the class that adopts this mixin:

    outputs : dict
    parameter_names : tuple
    discrepancy_name : str
    n_sim : int

    """

    @property
    def dim(self):
        """Return the number of parameters."""
        return len(self.parameter_names)

    @property
    def accept_rate(self):
        return min(1.0, self.n_samples / self.n_sim)

    @property
    def discrepancies(self):
        return self.outputs[self.discrepancy_name]

    @property
    def meta(self):
        return self.threshold, self.accept_rate, self.n_samples, self.n_sim

    @property
    def meta_names(self):
        return ('threshold', 'accept_rate', 'n_samples', 'n_sim')

    @property
    def n_samples(self):
        return len(self.discrepancies)

    @property
    def threshold(self):
        return np.min(self.discrepancies).item()


class Sample(OutputSampleMixin, ParameterInferenceResult):
    """Sampling results from inference methods."""

    def __init__(self,
                 outputs,
                 parameter_names,
                 n_sim,
                 method_name,
                 discrepancy_name,
                 weights=None
                 ):
        """Initialize result.

        Parameters
        ----------
        outputs : dict
            Dictionary with outputs from the nodes, e.g. samples.
        parameter_names : list
            Names of the parameter nodes
        n_sim : int
        method_name : string
            Name of inference method.
        discrepancy_name : string, optional
            Name of the discrepancy in outputs.
        weights : array_like
        **kwargs
            Other meta information for the result

        """
        super(Sample, self).__init__(outputs=outputs,
                                     parameter_names=parameter_names,
                                     n_sim=n_sim,
                                     method_name=method_name,
                                     )

        self.samples = OrderedDict()
        for n in self.parameter_names:
            self.samples[n] = self.outputs[n]

        self.discrepancy_name = discrepancy_name
        self.weights = weights

    @property
    def samples_array(self):
        """Return the samples as an array.

        The columns are in the same order as in self.parameter_names.

        Returns
        -------
        list of np.arrays

        """
        return np.column_stack(tuple(self.samples.values()))

    def __str__(self):
        """Return a summary of results as a string."""
        # create a buffer for capturing the output from summary's print statement
        stdout0 = sys.stdout
        buffer = io.StringIO()
        sys.stdout = buffer
        self.summary()
        sys.stdout = stdout0  # revert to original stdout
        return buffer.getvalue()

    def __repr__(self):
        """Return a summary of results as a string."""
        return self.__str__()

    def summary(self):
        """Print a verbose summary of contained results."""
        # TODO: include __str__ of Inference Task, seed?
        desc = "Method: {}\nNumber of samples: {}\n" \
            .format(self.method_name, self.n_samples)
        if hasattr(self, 'n_sim'):
            desc += "Number of simulations: {}\n".format(self.n_sim)
        if hasattr(self, 'threshold'):
            desc += "Threshold: {:.3g}\n".format(self.threshold)
        print(desc, end='')
        self.sample_means_summary()

    def sample_means_summary(self):
        """Print a representation of sample means."""
        s = "Sample means: "
        s += ', '.join(["{}: {:.3g}".format(k, v) for k, v in self.sample_means.items()])
        print(s)

    @property
    def sample_means(self):
        """Evaluate weighted averages of sampled parameters.

        Returns
        -------
        OrderedDict

        """
        return OrderedDict([(k, np.average(v, axis=0, weights=self.weights))
                            for k, v in self.samples.items()])

    @property
    def sample_means_array(self):
        """Evaluate weighted averages of sampled parameters.

        Returns
        -------
        np.array

        """
        return np.array(list(self.sample_means.values()))

    def plot_marginals(self, selector=None, bins=20, axes=None, **kwargs):
        """Plot marginal distributions for parameters.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional

        Returns
        -------
        axes : np.array of plt.Axes

        """
        return vis.plot_marginals(self.samples, selector, bins, axes, **kwargs)

    def plot_pairs(self, selector=None, bins=20, axes=None, **kwargs):
        """Plot pairwise relationships as a matrix with marginals on the diagonal.

        The y-axis of marginal histograms are scaled.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional

        Returns
        -------
        axes : np.array of plt.Axes

        """
        return vis.plot_pairs(self.samples, selector, bins, axes, **kwargs)


class SmcSample(Sample):
    """Container for results from SMC-ABC."""

    def __init__(self, method_name, outputs, parameter_names, populations, *args, **kwargs):
        """Initialize result.

        Parameters
        ----------
        method_name : str
        outputs : dict
        parameter_names : list
        populations : list[Sample]
            List of Sample objects
        args
        kwargs

        """
        super(SmcSample, self).__init__(
            method_name=method_name,
            outputs=outputs,
            parameter_names=parameter_names,
            *args,
            **kwargs)
        self.populations = populations

        if self.weights is None:
            raise ValueError("No weights provided for the sample")

    @property
    def n_populations(self):
        """Return the number of populations."""
        return len(self.populations)

    def summary(self, all=False):
        """Print a verbose summary of contained results.

        Parameters
        ----------
        all : bool, optional
            Whether to print the summary for all populations separately,
            or just the final population (default).

        """
        super(SmcSample, self).summary()

        if all:
            for i, pop in enumerate(self.populations):
                print('\nPopulation {}:'.format(i))
                pop.summary()

    def sample_means_summary(self, all=False):
        """Print a representation of sample means.

        Parameters
        ----------
        all : bool, optional
            Whether to print the means for all populations separately,
            or just the final population (default).

        """
        if all is False:
            super(SmcSample, self).sample_means_summary()
            return

        out = ''
        for i, pop in enumerate(self.populations):
            out += "Sample means for population {}: ".format(i)
            out += ', '.join(["{}: {:.3g}".format(k, v) for k, v in pop.sample_means.items()])
            out += '\n'
        print(out)

    def plot_marginals(self, selector=None, bins=20, axes=None, all=False, **kwargs):
        """Plot marginal distributions for parameters for all populations.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional
        all : bool, optional
            Plot the marginals of all populations

        """
        if all is False:
            super(SmcSample, self).plot_marginals()
            return

        fontsize = kwargs.pop('fontsize', 13)
        for i, pop in enumerate(self.populations):
            pop.plot_marginals(selector=selector, bins=bins, axes=axes)
            plt.suptitle("Population {}".format(i), fontsize=fontsize)

    def plot_pairs(self, selector=None, bins=20, axes=None, all=False, **kwargs):
        """Plot pairwise relationships as a matrix with marginals on the diagonal.

        The y-axis of marginal histograms are scaled.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional
        all : bool, optional
            Plot for all populations

        """
        if all is False:
            super(SmcSample, self).plot_marginals()
            return

        fontsize = kwargs.pop('fontsize', 13)
        for i, pop in enumerate(self.populations):
            pop.plot_pairs(selector=selector, bins=bins, axes=axes)
            plt.suptitle("Population {}".format(i), fontsize=fontsize)


class BolfiSample(Sample):
    """Container for results from BOLFI."""

    def __init__(self, method_name, chains, parameter_names, warmup, **kwargs):
        """Initialize result.

        Parameters
        ----------
        method_name : string
            Name of inference method.
        chains : np.array
            Chains from sampling, warmup included. Shape: (n_chains, n_samples, n_parameters).
        parameter_names : list : list of strings
            List of names in the outputs dict that refer to model parameters.
        warmup : int
            Number of warmup iterations in chains.

        """
        chains = chains.copy()
        shape = chains.shape
        n_chains = shape[0]
        warmed_up = chains[:, warmup:, :]
        concatenated = warmed_up.reshape((-1, ) + shape[2:])
        outputs = dict(zip(parameter_names, concatenated.T))

        super(BolfiSample, self).__init__(
            method_name=method_name,
            outputs=outputs,
            parameter_names=parameter_names,
            chains=chains,
            n_chains=n_chains,
            warmup=warmup,
            **kwargs)

    def plot_traces(self, selector=None, axes=None, **kwargs):
        """Plot MCMC traces."""
        return vis.plot_traces(self, selector, axes, **kwargs)


class OutputSampleCollector(OutputSampleMixin):
    """Collects outputs with the smallest threshold."""

    _index_key = '_index'

    def __init__(self, output_names, parameter_names, batch_size, max_sample_size,
                 discrepancy_name=None):
        """

        Parameters
        ----------
        output_names : tuple
        parameter_names : tuple
        batch_size : int
        max_sample_size : int
        discrepancy_name : str, optional
            Specify the discrepancy name in output_names. Default is the first name in
            output_names.

        """
        super(OutputSampleCollector, self).__init__()

        self.output_names = tuple(output_names) + (self._index_key,)
        self.parameter_names = tuple(parameter_names)
        self.discrepancy_name = discrepancy_name or self.output_names[0]
        self.batch_size = batch_size
        self.max_sample_size = max_sample_size

        self._outputs = {}
        self.n_batches = 0

    @property
    def n_sim(self):
        return self.n_batches * self.batch_size

    @property
    def outputs(self):
        return self.outputs_at(self.max_sample_size)

    def _init_outputs(self, batch):
        for output_name in self.output_names:

            if output_name not in batch:
                raise KeyError("Did not receive outputs for node {}".format(output_name))

            b = batch[output_name]
            if not is_array(b):
                raise ValueError('Output from node {} is not a numpy array. Please ensure that '
                                 'the corresponding node always returns numpy arrays.'
                                 .format(output_name))
            elif len(b) != self.batch_size:
                raise ValueError('Output from node {} returned outputs with length {}, '
                                 'but should have returned {} outputs (batch_size). Please ensure'
                                 'that the node returns a proper number of outputs'
                                 .format(output_name, len(b), self.batch_size))

            # Prepare the arrays
            shape = (self.max_sample_size + self.batch_size, ) + b.shape[1:]
            dtype = b.dtype
            self._outputs[output_name] = np.empty(shape, dtype=dtype)

            # Initialize distances with inf:s
            if output_name == self.discrepancy_name:
                self._outputs[output_name][:] = np.inf
            elif output_name == self._index_key:
                self._outputs[output_name][:] = np.nan

    def add_batch(self, batch, batch_index):
        """Collects outputs below current_threshold from the given batch.

        Parameters
        ----------
        batch : dict
        batch_index : int

        Returns
        -------
        None
        """

        # Add the indexes
        batch[self._index_key] = np.arange(batch_index,
                                           batch_index + self.batch_size,
                                           dtype=np.uint64)

        if not self._outputs:
            self._init_outputs(batch)

        for output_name, values in self._outputs.items():
            values[-self.batch_size:] = batch[output_name]

        sort_indices = np.argsort(self._outputs[self.discrepancy_name], axis=None)
        for values in self._outputs.values():
            values[:] = values[sort_indices]

        self.n_batches += 1

    def threshold_at(self, sample_size):
        return self.discrepancies[sample_size - 1].item()

    def accept_rate_at(self, sample_size):
        return min(1.0, sample_size / self.n_sim)

    def outputs_at(self, sample_size, sorted=True):
        """Return outputs for the given sample size.

        Parameters
        ----------
        sample_size : int, optional
            Default is the ``self.max_sample_size``.
        sorted : bool
            Return samples sorted by the threshold. Default is True. If false, will
            return a copy in the order they were sampled.

        Returns
        -------
        dict

        """
        if sample_size > self.max_sample_size:
            raise ValueError("The maximum sample size is {}, requested size was {}"
                             .format(self.max_sample_size, sample_size))

        outputs = {}
        for output_name, values in self._outputs.items():
            values_ = values[:sample_size]
            if sorted is False:
                values_ = values_.copy()
            outputs[output_name] = values_

        if sorted is False:
            # Put the samples in the original order
            sort_indices = np.argsort(outputs[self._index_key], axis=None)
            for values in outputs.values():
                values[:] = values[sort_indices]

        return outputs
