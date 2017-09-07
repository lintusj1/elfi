"""This module contains common inference methods."""

import logging
import warnings
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

import elfi.client
import elfi.methods.mcmc as mcmc
import elfi.visualization.interactive as visin
import elfi.visualization.visualization as vis
from elfi.loader import get_sub_seed
from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.utils import stochastic_optimization
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.results import BolfiSample, OptimizationResult, Sample, SmcSample, OutputSampleCollector
from elfi.methods.utils import (GMDistribution, ModelPrior, arr2d_to_batch,
                                batch_to_arr2d, ceil_to_batch_size, weighted_var)
from elfi.model.elfi_model import ComputationContext, ElfiModel, NodeReference

__all__ = ['Rejection', 'SMC', 'BayesianOptimization', 'BOLFI']


logger = logging.getLogger(__name__)


# TODO: refactor the plotting functions


class ParameterInference:
    """A base class for parameter inference methods.

    Attributes
    ----------
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm
    output_names : list
        Names of the nodes whose outputs are included in the batches
    client : elfi.client.ClientBase
        The batches are computed in the client
    max_parallel_batches : int
    state : dict
        Stores any changing data related to achieving the objective. Must include a key
        ``n_batches`` for determining when the inference is finished.
    objective : dict
        Holds the data for the algorithm to internally determine how many batches are still
        needed. You must have a key ``n_batches`` here. By default the algorithm finished when
        the ``n_batches`` in the state dictionary is equal or greater to the corresponding
        objective value.
    batch_handler : elfi.client.BatchHandler
        Helper class for submitting batches to the client and keeping track of their
        indexes.
    pool : elfi.store.OutputPool
        Pool object for storing and reusing node outputs.


    """

    def __init__(self,
                 model,
                 output_names,
                 batch_size=1000,
                 seed=None,
                 pool=None,
                 max_parallel_batches=None):
        """Construct the inference algorithm object.

        If you are implementing your own algorithm do not forget to call `super`.

        Parameters
        ----------
        model : ElfiModel
            Model to perform the inference with.
        output_names : list
            Names of the nodes whose outputs will be requested from the ELFI graph.
        batch_size : int, optional
        seed : int, optional
            Seed for the data generation from the ElfiModel
        pool : OutputPool, optional
            OutputPool both stores and provides precomputed values for batches.
        max_parallel_batches : int, optional
            Maximum number of batches allowed to be in computation at the same time.
            Defaults to number of cores in the client


        """
        model = model.model if isinstance(model, NodeReference) else model
        if not model.parameter_names:
            raise ValueError('Model {} defines no parameters'.format(model))

        self.model = model.copy()
        self.output_names = self._check_outputs(output_names)
        self.client = elfi.client.get_client()
        self.context = ComputationContext(batch_size=batch_size, seed=seed, pool=pool)
        self.batch_handler = elfi.client.BatchHandler(self.model,
                                                      context=self.context,
                                                      output_names=output_names,
                                                      client=self.client)
        self.max_parallel_batches = max_parallel_batches or self.client.num_cores or 1
        self.n_sim = 0

        if self.max_parallel_batches < 1:
            raise ValueError('The value of max_parallel_batches must be at least 1.')
        if self.client.num_cores < 1:
            warnings.warn('The client has currently no workers available. Please ensure that it '
                          'will receive workers before continuing')

    @property
    def pool(self):
        """Return the output pool of the inference."""
        return self.context.pool

    @property
    def seed(self):
        """Return the seed of the inference."""
        return self.context.seed

    @property
    def parameter_names(self):
        """Return the parameters to be inferred."""
        return self.model.parameter_names

    @property
    def batch_size(self):
        """Return the current batch_size."""
        return self.context.batch_size

    def iterate(self, total_limit=None):
        """Advance the inference by one iteration.

        This is a way to manually progress the inference. One iteration consists of
        waiting and processing the result of the next batch in succession and possibly
        submitting new batches.

        Notes
        -----
        If the next batch is ready, it will be processed immediately and no new batches
        are submitted.

        New batches are submitted only while waiting for the next one to complete. There
        will never be more batches submitted in parallel than the `max_parallel_batches`
        setting allows.

        Returns
        -------
        bool
            Was a new batch processed?


        """
        total_limit = total_limit or np.inf
        if self.n_sim >= total_limit:
            return False

        self.submit(total_limit)

        if self.batch_handler.num_pending == 0:
            # Basically we should not end up here
            logger.warning('No batches to process in the queue. Please check max_parallel_batches '
                           'is positive.')
            return False

        # Wait and handle the next ready batch in succession
        batch, batch_index = self.batch_handler.wait_next()

        logger.info('Processing batch %d' % batch_index)
        self.update(batch, batch_index)

        return True

    def extract(self):
        """Prepare the result from the current state of the inference.

        ELFI calls this method in the end of the inference to return the result.

        Returns
        -------
        result : elfi.methods.result.Result

        """
        raise NotImplementedError

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        ELFI calls this method when a new batch has been computed and the state of
        the inference should be updated with it. It is also possible to bypass ELFI and
        call this directly to update the inference.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        Returns
        -------
        None

        """
        self.n_sim += self.batch_size

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        ELFI calls this method before submitting a new batch with an increasing index
        `batch_index`. This is an optional method to override. Use this if you have a need
        do do preparations, e.g. in Bayesian optimization algorithm, the next acquisition
        points would be acquired here.

        If you need provide values for certain nodes, you can do so by constructing a
        batch dictionary and returning it. See e.g. BayesianOptimization for an example.

        Parameters
        ----------
        batch_index : int
            next batch_index to be submitted

        Returns
        -------
        batch : dict or None
            Keys should match to node names in the model. These values will override any
            default values or operations in those nodes.

        """
        pass

    def submit(self, total_limit=None):
        # Submit as many new batches as allowed but at most up to n_sim simulations
        n_submits = 0

        while self._allow_submit(self.batch_handler.next_index, total_limit):
            logger.info("Submitting batch %d" % self.batch_handler.next_index)

            next_batch = self.prepare_new_batch(self.batch_handler.next_index)
            self.batch_handler.submit(next_batch)
            n_submits += 1

        return n_submits

    def plot_state(self, **kwargs):
        """Plot the current state of the algorithm.

        Parameters
        ----------
        axes : matplotlib.axes.Axes (optional)
        figure : matplotlib.figure.Figure (optional)
        xlim
            x-axis limits
        ylim
            y-axis limits
        interactive : bool (default False)
            If true, uses IPython.display to update the cell figure
        close
            Close figure in the end of plotting. Used in the end of interactive mode.

        Returns
        -------
        None

        """
        raise NotImplementedError

    def _allow_submit(self, batch_index, total_limit=None):
        total_limit = total_limit or np.inf
        n_pending = self.batch_handler.num_pending

        allow = True
        allow &= self.max_parallel_batches > n_pending
        allow &= total_limit > self.n_sim + n_pending
        allow &= not self.batch_handler.has_ready()
        return allow

    @staticmethod
    def _resolve_model(model, target, default_reference_class=NodeReference):
        if isinstance(model, ElfiModel) and target is None:
            raise NotImplementedError("Please specify the target node of the inference method")

        if isinstance(model, NodeReference):
            target = model
            model = target.model

        if isinstance(target, str):
            target = model[target]

        if not isinstance(target, default_reference_class):
            raise ValueError('Unknown target node class')

        return model, target.name

    def _check_outputs(self, output_names):
        """Filter out duplicates and check that corresponding nodes exist.

        Preserves the order.
        """
        output_names = output_names or []
        checked_names = []
        seen = set()
        for name in output_names:
            if isinstance(name, NodeReference):
                name = name.name

            if name in seen:
                continue
            elif not isinstance(name, str):
                raise ValueError(
                    'All output names must be strings, object {} was given'.format(name))
            elif not self.model.has_node(name):
                raise ValueError('Node {} output was requested, but it is not in the model.')

            seen.add(name)
            checked_names.append(name)

        return checked_names


class ABCSampler(ParameterInference):
    pass


class Rejection(ABCSampler):
    """Parallel ABC rejection sampler.

    For a description of the rejection sampler and a general introduction to ABC, see e.g.
    Lintusaari et al. 2016.

    References
    ----------
    Lintusaari J, Gutmann M U, Dutta R, Kaski S, Corander J (2016). Fundamentals and
    Recent Developments in Approximate Bayesian Computation. Systematic Biology.
    http://dx.doi.org/10.1093/sysbio/syw077.

    """

    def __init__(self, model, discrepancy_name=None, output_names=None, sample_sizes=None,
                 max_sample_size=None, **kwargs):
        """Initialize the Rejection sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        sample_sizes : list[int], optional
            Sample sizes that we wish to track the threshold for. Calling the ``threshold`` method
            will return a tuple listing the thresholds for the respective sample sizes. Default
            is [1, max_sample_size].
        max_sample_size : int, optional
            The maximum sample size that can be extracted. Larger value requires more memory.
            Default is the maximum among 10000, batch_size and max(sample_sizes).
        kwargs:
            See InferenceMethod

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        output_names = [discrepancy_name] + model.parameter_names + (output_names or [])

        super(Rejection, self).__init__(model, output_names, **kwargs)

        self.discrepancy_name = discrepancy_name
        self.max_sample_size = max(self.batch_size, max(sample_sizes or [1000]))
        self.sample_sizes = sample_sizes or [1, self.max_sample_size]  # type: list
        self.sample_points = OutputSampleCollector(self.output_names,
                                                   self.parameter_names,
                                                   self.batch_size,
                                                   max_sample_size=self.max_sample_size)

    def sample(self, n_samples, threshold=None, quantile=None, n_sim=None):
        """Set objective for inference.

        Parameters
        ----------
        n_samples : int
            number of samples to generate
        threshold : float
            Acceptance threshold
        quantile : float
            In between (0,1). Define the threshold as the p-quantile of all the
            simulations. n_sim = n_samples/quantile.
        n_sim : int
            Total number of simulations. The threshold will be the n_samples smallest
            discrepancy among n_sim simulations.

        """
        if threshold is not None:
            n_sim = self.max_parallel_batches*self.batch_size
        elif quantile is not None:
            threshold = np.inf
            n_sim = ceil(n_samples / quantile)
        elif n_sim is not None:
            threshold = np.inf
        else:
            threshold = np.inf
            # Corresponds to a 0.01 quantile
            n_sim = n_samples * 100

        while self.iterate(n_sim):
            # If threshold, keep simulating until the threshold is reached
            if threshold < self.sample_points.threshold_at(n_samples):
                n_sim += self.batch_size

        return self.extract(n_samples)

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(Rejection, self).update(batch, batch_index)
        self.sample_points.add_batch(batch, batch_index)

    @property
    def thresholds(self):
        """Returns the thresholds for ``self.sample_sizes``."""
        return tuple([self.sample_points.threshold_at(s) for s in self.sample_sizes])

    def extract(self, n_samples=None):
        """Extracts a sample with the current threshold.

        Returns
        -------
        result : Sample

        """
        n_samples = n_samples or self.max_sample_size

        return Sample(outputs=self.sample_points.outputs_at(n_samples, sorted=False),
                      parameter_names=self.parameter_names,
                      n_sim=self.n_sim,
                      method_name=self.__class__.__name__,
                      discrepancy_name=self.discrepancy_name,
                      )

    def plot_state(self, **options):
        """Plot the current state of the inference algorithm.

        This feature is still experimental and only supports 1d or 2d cases.
        """
        n_samples = self.sample_sizes[-1]

        displays = []
        if options.get('interactive'):
            from IPython import display
            displays.append(display.HTML('<span>Thresholds: {}</span>'.
                                         format(self.thresholds)))

        visin.plot_sample(
            self.sample_points.outputs,
            nodes=self.parameter_names,
            n=n_samples,
            displays=displays,
            **options)


class SMC(ABCSampler):
    """Sequential Monte Carlo ABC sampler."""

    def __init__(self, model, discrepancy_name=None, output_names=None, **kwargs):
        """Initialize the SMC-ABC sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        kwargs:
            See InferenceMethod

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)

        super(SMC, self).__init__(model, output_names, **kwargs)

        self._prior = ModelPrior(self.model)
        self.discrepancy_name = discrepancy_name
        self.state['round'] = 0
        self._populations = []
        self._rejection = None
        self._round_random_state = None

    def set_objective(self, n_samples, thresholds):
        """Set the objective of the inference."""
        self.objective.update(
            dict(
                n_samples=n_samples,
                n_batches=self.max_parallel_batches,
                round=len(thresholds) - 1,
                thresholds=thresholds))
        self._init_new_round()

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        SmcSample

        """
        # Extract information from the population
        pop = self._extract_population()
        return SmcSample(
            outputs=pop.outputs,
            populations=self._populations.copy() + [pop],
            weights=pop.weights,
            threshold=pop.threshold,
            **self._extract_result_kwargs())

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(SMC, self).update(batch, batch_index)
        self._rejection.update(batch, batch_index)

        if self._rejection.finished:
            self.batch_handler.cancel_pending()
            if self.state['round'] < self.objective['round']:
                self._populations.append(self._extract_population())
                self.state['round'] += 1
                self._init_new_round()

        self._update_objective()

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        Parameters
        ----------
        batch_index : int
            next batch_index to be submitted

        Returns
        -------
        batch : dict or None
            Keys should match to node names in the model. These values will override any
            default values or operations in those nodes.

        """
        if self.state['round'] == 0:
            # Use the actual prior
            return

        # Sample from the proposal, condition on actual prior
        params = GMDistribution.rvs(*self._gm_params, size=self.batch_size,
                                    prior_logpdf=self._prior.logpdf,
                                    random_state=self._round_random_state)

        batch = arr2d_to_batch(params, self.parameter_names)
        return batch

    def _init_new_round(self):
        round = self.state['round']

        dashes = '-' * 16
        logger.info('%s Starting round %d %s' % (dashes, round, dashes))

        # Get a subseed for this round for ensuring consistent results for the round
        seed = self.seed if round == 0 else get_sub_seed(self.seed, round)
        self._round_random_state = np.random.RandomState(seed)

        self._rejection = Rejection(
            self.model,
            discrepancy_name=self.discrepancy_name,
            output_names=self.output_names,
            batch_size=self.batch_size,
            seed=seed,
            max_parallel_batches=self.max_parallel_batches)

        self._rejection.set_objective(
            self.objective['n_samples'], threshold=self.current_population_threshold)

    def _extract_population(self):
        sample = self._rejection.extract_result()
        # Append the sample object
        sample.method_name = "Rejection within SMC-ABC"
        w, cov = self._compute_weights_and_cov(sample)
        sample.weights = w
        sample.meta['cov'] = cov
        return sample

    def _compute_weights_and_cov(self, pop):
        params = np.column_stack(tuple([pop.outputs[p] for p in self.parameter_names]))

        if self._populations:
            q_logpdf = GMDistribution.logpdf(params, *self._gm_params)
            p_logpdf = self._prior.logpdf(params)
            w = np.exp(p_logpdf - q_logpdf)
        else:
            w = np.ones(pop.n_samples)

        if np.count_nonzero(w) == 0:
            raise RuntimeError("All sample weights are zero. If you are using a prior "
                               "with a bounded support, this may be caused by specifying "
                               "a too small sample size.")

        # New covariance
        cov = 2 * np.diag(weighted_var(params, w))

        if not np.all(np.isfinite(cov)):
            logger.warning("Could not estimate the sample covariance. This is often "
                           "caused by majority of the sample weights becoming zero."
                           "Falling back to using unit covariance.")
            cov = np.diag(np.ones(params.shape[1]))

        return w, cov

    def _update_objective(self):
        """Update the objective n_batches."""
        n_batches = sum([pop.n_batches for pop in self._populations])
        self.objective['n_batches'] = n_batches + self._rejection.objective['n_batches']

    @property
    def _gm_params(self):
        sample = self._populations[-1]
        params = sample.samples_array
        return params, sample.cov, sample.weights

    @property
    def current_population_threshold(self):
        """Return the threshold for current population."""
        return self.objective['thresholds'][self.state['round']]


class BayesianOptimization(ParameterInference):
    """Bayesian Optimization of an unknown target function."""

    def __init__(self,
                 model,
                 target_name=None,
                 bounds=None,
                 initial_evidence=None,
                 update_interval=10,
                 target_model=None,
                 acquisition_method=None,
                 acq_noise_var=0,
                 exploration_rate=10,
                 batch_size=1,
                 batches_per_acquisition=None,
                 async=False,
                 **kwargs):
        """Initialize Bayesian optimization.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        target_name : str or NodeReference
            Only needed if model is an ElfiModel
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters: dict('parameter_name':(lower, upper), ... )`. Not used if
            custom target_model is given.
        initial_evidence : int, dict, optional
            Number of initial evidence or a precomputed batch dict containing parameter
            and discrepancy values. Default value depends on the dimensionality.
        update_interval : int, optional
            How often to update the GP hyperparameters of the target_model
        target_model : GPyRegression, optional
        acquisition_method : Acquisition, optional
            Method of acquiring evidence points. Defaults to LCBSC.
        acq_noise_var : float or np.array, optional
            Variance(s) of the noise added in the default LCBSC acquisition method.
            If an array, should be 1d specifying the variance for each dimension.
        exploration_rate : float, optional
            Exploration rate of the acquisition method
        batch_size : int, optional
            Elfi batch size. Defaults to 1.
        batches_per_acquisition : int, optional
            How many batches will be requested from the acquisition function at one go.
            Defaults to max_parallel_batches.
        async : bool, optional
            Allow acquisitions to be made asynchronously, i.e. do not wait for all the
            results from the previous acquisition before making the next. This can be more
            efficient with a large amount of workers (e.g. in cluster environments) but
            forgoes the guarantee for the exactly same result with the same initial
            conditions (e.g. the seed). Default False.
        **kwargs

        """
        model, target_name = self._resolve_model(model, target_name)
        output_names = [target_name] + model.parameter_names
        super(BayesianOptimization, self).__init__(
            model, output_names, batch_size=batch_size, **kwargs)

        target_model = target_model or \
            GPyRegression(self.model.parameter_names, bounds=bounds)

        self.target_name = target_name
        self.target_model = target_model

        n_precomputed = 0
        n_initial, precomputed = self._resolve_initial_evidence(initial_evidence)
        if precomputed is not None:
            params = batch_to_arr2d(precomputed, self.parameter_names)
            n_precomputed = len(params)
            self.target_model.update(params, precomputed[target_name])

        self.batches_per_acquisition = batches_per_acquisition or self.max_parallel_batches
        self.acquisition_method = acquisition_method or \
            LCBSC(self.target_model,
                  prior=ModelPrior(self.model),
                  noise_var=acq_noise_var,
                  exploration_rate=exploration_rate,
                  seed=self.seed)

        self.n_initial_evidence = n_initial
        self.n_precomputed_evidence = n_precomputed
        self.update_interval = update_interval
        self.async = async

        self.state['n_evidence'] = self.n_precomputed_evidence
        self.state['last_GP_update'] = self.n_initial_evidence
        self.state['acquisition'] = []

    def _resolve_initial_evidence(self, initial_evidence):
        # Some sensibility limit for starting GP regression
        precomputed = None
        n_required = max(10, 2**self.target_model.input_dim + 1)
        n_required = ceil_to_batch_size(n_required, self.batch_size)

        if initial_evidence is None:
            n_initial_evidence = n_required
        elif isinstance(initial_evidence, (int, np.int, float)):
            n_initial_evidence = int(initial_evidence)
        else:
            precomputed = initial_evidence
            n_initial_evidence = len(precomputed[self.target_name])

        if n_initial_evidence < 0:
            raise ValueError('Number of initial evidence must be positive or zero '
                             '(was {})'.format(initial_evidence))
        elif n_initial_evidence < n_required:
            logger.warning('We recommend having at least {} initialization points for '
                           'the initialization (now {})'.format(n_required, n_initial_evidence))

        if precomputed is None and (n_initial_evidence % self.batch_size != 0):
            logger.warning('Number of initial_evidence %d is not divisible by '
                           'batch_size %d. Rounding it up...' % (n_initial_evidence,
                                                                 self.batch_size))
            n_initial_evidence = ceil_to_batch_size(n_initial_evidence, self.batch_size)

        return n_initial_evidence, precomputed

    @property
    def n_evidence(self):
        """Return the number of acquired evidence points."""
        return self.state.get('n_evidence', 0)

    @property
    def acq_batch_size(self):
        """Return the total number of acquisition per iteration."""
        return self.batch_size * self.batches_per_acquisition

    def set_objective(self, n_evidence=None):
        """Set objective for inference.

        You can continue BO by giving a larger n_evidence.

        Parameters
        ----------
        n_evidence : int
            Number of total evidence for the GP fitting. This includes any initial
            evidence.

        """
        if n_evidence is None:
            n_evidence = self.objective.get('n_evidence', self.n_evidence)

        if n_evidence < self.n_evidence:
            logger.warning('Requesting less evidence than there already exists')

        self.objective['n_evidence'] = n_evidence
        self.objective['n_sim'] = n_evidence - self.n_precomputed_evidence

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        OptimizationResult

        """
        x_min, _ = stochastic_optimization(
            self.target_model.predict_mean, self.target_model.bounds, seed=self.seed)

        batch_min = arr2d_to_batch(x_min, self.parameter_names)
        outputs = arr2d_to_batch(self.target_model.X, self.parameter_names)
        outputs[self.target_name] = self.target_model.Y

        return OptimizationResult(
            x_min=batch_min, outputs=outputs, **self._extract_result_kwargs())

    def update(self, batch, batch_index):
        """Update the GP regression model of the target node with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(BayesianOptimization, self).update(batch, batch_index)
        self.state['n_evidence'] += self.batch_size

        params = batch_to_arr2d(batch, self.parameter_names)
        self._report_batch(batch_index, params, batch[self.target_name])

        optimize = self._should_optimize()
        self.target_model.update(params, batch[self.target_name], optimize)
        if optimize:
            self.state['last_GP_update'] = self.target_model.n_evidence

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        Parameters
        ----------
        batch_index : int
            next batch_index to be submitted

        Returns
        -------
        batch : dict or None
            Keys should match to node names in the model. These values will override any
            default values or operations in those nodes.

        """
        t = self._get_acquisition_index(batch_index)

        # Check if we still should take initial points from the prior
        if t < 0:
            return

        # Take the next batch from the acquisition_batch
        acquisition = self.state['acquisition']
        if len(acquisition) == 0:
            acquisition = self.acquisition_method.acquire(self.acq_batch_size, t=t)

        batch = arr2d_to_batch(acquisition[:self.batch_size], self.parameter_names)
        self.state['acquisition'] = acquisition[self.batch_size:]

        return batch

    def _get_acquisition_index(self, batch_index):
        acq_batch_size = self.batch_size * self.batches_per_acquisition
        initial_offset = self.n_initial_evidence - self.n_precomputed_evidence
        starting_sim_index = self.batch_size * batch_index

        t = (starting_sim_index - initial_offset) // acq_batch_size
        return t

    # TODO: use state dict
    @property
    def _n_submitted_evidence(self):
        return self.batch_handler.total * self.batch_size

    def _allow_submit(self, batch_index):
        if not super(BayesianOptimization, self)._allow_submit(batch_index):
            return False

        if self.async:
            return True

        # Allow submitting freely as long we are still submitting initial evidence
        t = self._get_acquisition_index(batch_index)
        if t < 0:
            return True

        # Do not allow acquisition until previous acquisitions are ready (as well
        # as all initial acquisitions)
        acquisitions_left = len(self.state['acquisition'])
        if acquisitions_left == 0 and self.batch_handler.has_pending:
            return False

        return True

    def _should_optimize(self):
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_GP_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update

    def _report_batch(self, batch_index, params, distances):
        str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        for i in range(self.batch_size):
            str += "{}{} at {}\n".format(fill, distances[i].item(), params[i])
        logger.debug(str)

    def plot_state(self, **options):
        """Plot the GP surface.

        This feature is still experimental and currently supports only 2D cases.
        """
        f = plt.gcf()
        if len(f.axes) < 2:
            f, _ = plt.subplots(1, 2, figsize=(13, 6), sharex='row', sharey='row')

        gp = self.target_model

        # Draw the GP surface
        visin.draw_contour(
            gp.predict_mean,
            gp.bounds,
            self.parameter_names,
            title='GP target surface',
            points=gp.X,
            axes=f.axes[0],
            **options)

        # Draw the latest acquisitions
        if options.get('interactive'):
            point = gp.X[-1, :]
            if len(gp.X) > 1:
                f.axes[1].scatter(*point, color='red')

        displays = [gp._gp]

        if options.get('interactive'):
            from IPython import display
            displays.insert(
                0,
                display.HTML('<span><b>Iteration {}:</b> Acquired {} at {}</span>'.format(
                    len(gp.Y), gp.Y[-1][0], point)))

        # Update
        visin._update_interactive(displays, options)

        def acq(x):
            return self.acquisition_method.evaluate(x, len(gp.X))

        # Draw the acquisition surface
        visin.draw_contour(
            acq,
            gp.bounds,
            self.parameter_names,
            title='Acquisition surface',
            points=None,
            axes=f.axes[1],
            **options)

        if options.get('close'):
            plt.close()

    def plot_discrepancy(self, axes=None, **kwargs):
        """Plot acquired parameters vs. resulting discrepancy.

        TODO: refactor
        """
        n_plots = self.target_model.input_dim
        ncols = kwargs.pop('ncols', 5)
        kwargs['sharey'] = kwargs.get('sharey', True)
        shape = (max(1, n_plots // ncols), min(n_plots, ncols))
        axes, kwargs = vis._create_axes(axes, shape, **kwargs)
        axes = axes.ravel()

        for ii in range(n_plots):
            axes[ii].scatter(self.target_model._gp.X[:, ii], self.target_model._gp.Y[:, 0])
            axes[ii].set_xlabel(self.parameter_names[ii])

        axes[0].set_ylabel('Discrepancy')

        return axes


class BOLFI(BayesianOptimization):
    """Bayesian Optimization for Likelihood-Free Inference (BOLFI).

    Approximates the discrepancy function by a stochastic regression model.
    Discrepancy model is fit by sampling the discrepancy function at points decided by
    the acquisition function.

    The method implements the framework introduced in Gutmann & Corander, 2016.

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    """

    def fit(self, n_evidence, threshold=None):
        """Fit the surrogate model.

        Generates a regression model for the discrepancy given the parameters.

        Currently only Gaussian processes are supported as surrogate models.

        Parameters
        ----------
        threshold : float, optional
            Discrepancy threshold for creating the posterior (log with log discrepancy).

        """
        logger.info("BOLFI: Fitting the surrogate model...")

        if n_evidence is None:
            raise ValueError(
                'You must specify the number of evidence (n_evidence) for the fitting')

        self.infer(n_evidence)
        return self.extract_posterior(threshold)

    def extract_posterior(self, threshold=None):
        """Return an object representing the approximate posterior.

        The approximation is based on surrogate model regression.

        Parameters
        ----------
        threshold: float, optional
            Discrepancy threshold for creating the posterior (log with log discrepancy).

        Returns
        -------
        posterior : elfi.methods.posteriors.BolfiPosterior

        """
        if self.state['n_batches'] == 0:
            raise ValueError('Model is not fitted yet, please see the `fit` method.')

        return BolfiPosterior(self.target_model, threshold=threshold, prior=ModelPrior(self.model))

    def sample(self,
               n_samples,
               warmup=None,
               n_chains=4,
               threshold=None,
               initials=None,
               algorithm='nuts',
               n_evidence=None,
               **kwargs):
        r"""Sample the posterior distribution of BOLFI.

        Here the likelihood is defined through the cumulative density function
        of the standard normal distribution:

        L(\theta) \propto F((h-\mu(\theta)) / \sigma(\theta))

        where h is the threshold, and \mu(\theta) and \sigma(\theta) are the posterior mean and
        (noisy) standard deviation of the associated Gaussian process.

        The sampling is performed with an MCMC sampler (the No-U-Turn Sampler, NUTS).

        Parameters
        ----------
        n_samples : int
            Number of requested samples from the posterior for each chain. This includes warmup,
            and note that the effective sample size is usually considerably smaller.
        warmpup : int, optional
            Length of warmup sequence in MCMC sampling. Defaults to n_samples//2.
        n_chains : int, optional
            Number of independent chains.
        threshold : float, optional
            The threshold (bandwidth) for posterior (give as log if log discrepancy).
        initials : np.array of shape (n_chains, n_params), optional
            Initial values for the sampled parameters for each chain.
            Defaults to best evidence points.
        algorithm : string, optional
            Sampling algorithm to use. Currently only 'nuts' is supported.
        n_evidence : int
            If the regression model is not fitted yet, specify the amount of evidence

        Returns
        -------
        BolfiSample

        """
        if self.state['n_batches'] == 0:
            self.fit(n_evidence)

        # TODO: other MCMC algorithms

        posterior = self.extract_posterior(threshold)
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with smallest discrepancy
        if initials is not None:
            if np.asarray(initials).shape != (n_chains, self.target_model.input_dim):
                raise ValueError("The shape of initials must be (n_chains, n_params).")
        else:
            inds = np.argsort(self.target_model.Y[:, 0])
            initials = np.asarray(self.target_model.X[inds])

        self.target_model.is_sampling = True  # enables caching for default RBF kernel

        tasks_ids = []
        ii_initial = 0

        # sampling is embarrassingly parallel, so depending on self.client this may parallelize
        for ii in range(n_chains):
            seed = get_sub_seed(self.seed, ii)
            # discard bad initialization points
            while np.isinf(posterior.logpdf(initials[ii_initial])):
                ii_initial += 1
                if ii_initial == len(inds):
                    raise ValueError(
                        "BOLFI.sample: Cannot find enough acceptable initialization points!")

            tasks_ids.append(
                self.client.apply(
                    mcmc.nuts,
                    n_samples,
                    initials[ii_initial],
                    posterior.logpdf,
                    posterior.gradient_logpdf,
                    n_adapt=warmup,
                    seed=seed,
                    **kwargs))
            ii_initial += 1

        # get results from completed tasks or run sampling (client-specific)
        chains = []
        for id in tasks_ids:
            chains.append(self.client.get_result(id))

        chains = np.asarray(chains)

        print(
            "{} chains of {} iterations acquired. Effective sample size and Rhat for each "
            "parameter:".format(n_chains, n_samples))
        for ii, node in enumerate(self.parameter_names):
            print(node, mcmc.eff_sample_size(chains[:, :, ii]),
                  mcmc.gelman_rubin(chains[:, :, ii]))

        self.target_model.is_sampling = False

        return BolfiSample(
            method_name='BOLFI',
            chains=chains,
            parameter_names=self.parameter_names,
            warmup=warmup,
            threshold=float(posterior.threshold),
            n_sim=self.state['n_sim'],
            seed=self.seed)
