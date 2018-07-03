.. pysf documentation master file, created by
   sphinx-quickstart on Tue Jul  3 11:17:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pysf's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Data container
=======================================
   
.. autoclass:: pysf.data.MultiSeries   
   :members:
   :undoc-members:
.. COMMENTEDOUT   :private-members:
.. COMMENTEDOUT   :special-members:

Framework for predictors & transformers
=======================================
   
.. autoclass:: pysf.predictors.framework.AbstractPredictor
   :members:
   :undoc-members:
   :private-members:
.. COMMENTEDOUT   :special-members:

.. autoclass:: pysf.predictors.framework.PipelinePredictor
   :members:
   :undoc-members:
   :private-members:
.. COMMENTEDOUT   :special-members:
   
.. autoclass:: pysf.errors.RawResiduals
   :members:
   :undoc-members:
.. COMMENTEDOUT   :private-members:
.. COMMENTEDOUT   :special-members:
   
.. autoclass:: pysf.errors.ErrorCurve
   :members:
   :undoc-members:
.. COMMENTEDOUT   :private-members:
.. COMMENTEDOUT   :special-members:
   
.. autoclass:: pysf.predictors.framework.ScoringResult
   :members:
   :undoc-members:
   :private-members:
.. COMMENTEDOUT   :special-members:

.. autoclass:: pysf.transformers.framework.AbstractTransformer
   :members:
   :undoc-members:
   :private-members:
.. COMMENTEDOUT   :special-members:



Self-tuning predictors
=======================================
   
.. autoclass:: pysf.predictors.tuning.TuningMetrics
   :members:
   :undoc-members:
   :private-members:
.. COMMENTEDOUT   :special-members:

.. autoclass:: pysf.predictors.tuning.TuningOverallPredictor
   :members:
   :undoc-members:
.. COMMENTEDOUT   :private-members:
.. COMMENTEDOUT   :special-members:

.. autoclass:: pysf.predictors.tuning.TuningTimestampMultiplexerPredictor
   :members:
   :undoc-members:
.. COMMENTEDOUT   :private-members:
.. COMMENTEDOUT   :special-members:
   

All predictors & transformers
=======================================   

# TODO: Add this in later
#.. inheritance-diagram:: pysf.predictors.tuning.TuningOverallPredictor  pysf.predictors.tuning.TuningTimestampMultiplexerPredictor pysf.predictors.forest.MultiCurveRandomForestPredictor pysf.predictors.kernels.MultiCurveKernelsPredictor pysf.predictors.framework.SingleCurveSeriesPredictor pysf.predictors.framework.MultiCurveTabularPredictor pysf.predictors.framework.MultiCurveTabularWindowedPredictor pysf.predictors.framework.SingleCurveTabularWindowedPredictor pysf.predictors.baselines.SeriesLinearInterpolator pysf.predictors.baselines.ZeroPredictor pysf.predictors.baselines.SeriesMeansPredictor pysf.predictors.baselines.TimestampMeansPredictor pysf.transformers.framework.AbstractTransformer pysf.transformers.smoothing.SmoothingSplineTransformer pysf.predictors.lstm.MultiCurveWindowedLstmPredictor
   
Generalisation error estimation
=======================================
   
.. autoclass:: pysf.generalisation.Target   
   :members:
   :undoc-members:
   :private-members:
.. COMMENTEDOUT   :special-members:

.. autoclass:: pysf.generalisation.GeneralisationPerformanceEvaluator
   :members:
   :undoc-members:
   :private-members:
.. COMMENTEDOUT   :special-members:
   
.. (This is a comment.) This line aims to automatically document everything in pysf:
.. automodule:: pysf
   :members:
   :undoc-members:
   :private-members:
   :special-members:
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
