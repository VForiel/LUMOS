# API reference

```{eval-rst}
.. automodule:: lumos

.. autofunction:: lumos.hardware_report

.. autofunction:: lumos.jit
```

## Units

```{eval-rst}
.. automodule:: lumos.units
   :members:
```

## Configuration and backends

```{eval-rst}
``lumos.config`` is the instance of :class:`lumos._config.Config` holding the settings of the process.

.. autoclass:: lumos._config.Config
   :members:

.. automodule:: lumos.backend
   :members: Backend, BackendUnavailableError, available, known, get, current, use, get_default, set_default, backend_of, to_numpy, convert

.. automodule:: lumos.backend.fft
   :members:

.. automodule:: lumos.backend.hardware
   :members:

.. automodule:: lumos.backend.memory
   :members:
```

## Core

```{eval-rst}
.. automodule:: lumos.core.param
   :members: Param, State, Setting, ParamHost

.. automodule:: lumos.core.priors
   :members:

.. automodule:: lumos.core.signal
   :members:

.. automodule:: lumos.core.port
   :members:

.. automodule:: lumos.core.element
   :members:

.. automodule:: lumos.core.registry
   :members: Registry, UnknownTypeError, registry, register
```

## Fields

```{eval-rst}
.. automodule:: lumos.fields.grid
   :members: Grid

.. automodule:: lumos.fields.wavefront
   :members: Wavefront

.. automodule:: lumos.fields.wavecube
   :members: Wavecube

.. automodule:: lumos.fields.incoherent
   :members: IncoherentBundle

.. automodule:: lumos.fields.radiance
   :members: RadianceField, CANONICAL_RADIANCE

.. automodule:: lumos.fields.modal
   :members: ModalField

.. automodule:: lumos.fields.dataproduct
   :members: DataProduct

.. automodule:: lumos.fields.smatrix
   :members:

.. automodule:: lumos.fields.spectral
   :members:

.. automodule:: lumos.fields.jones
   :members:
```

## Contexts and graphs

```{eval-rst}
.. autoclass:: lumos.Context
   :members:

.. automodule:: lumos.core.context
   :members: Observation, SweepPoint

.. automodule:: lumos.core.graph
   :members: Graph, Connection, Result, GraphError, CycleError

.. automodule:: lumos.core.group
   :members: ElementGroup

.. automodule:: lumos.core.clock
   :members: Clock

.. automodule:: lumos.core.action
   :members: action, ActionSpec

.. automodule:: lumos.core.driver
   :members: Driver, SimulatedDriver, Command, Reply, DriverError

.. automodule:: lumos.core.journal
   :members: Journal, JournalEntry, JsonFormatter
```

## Files

```{eval-rst}
.. automodule:: lumos.io
   :members:

.. automodule:: lumos.io.codegen
   :members: to_python

.. automodule:: lumos.errors
   :members:
```

## Propagation

```{eval-rst}
.. automodule:: lumos.propagation
   :members: propagate, focus, plan, methods, register_method, cache_info, clear_cache

.. automodule:: lumos.propagation.planner
   :members: Plan

.. automodule:: lumos.propagation.rs
.. automodule:: lumos.propagation.fraunhofer
.. automodule:: lumos.propagation.fresnel
.. automodule:: lumos.propagation.asm
```

## Plotting

```{eval-rst}
.. automodule:: lumos.plotting
   :members:
```
