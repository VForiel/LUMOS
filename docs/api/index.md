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

## Physical models

```{eval-rst}
.. automodule:: lumos.physics.blackbody
   :members:

.. automodule:: lumos.physics.photometry
   :members:

.. automodule:: lumos.physics.orbits
   :members:

.. automodule:: lumos.physics.apertures
   :members:

.. automodule:: lumos.physics.zernike
   :members:

.. automodule:: lumos.physics.turbulence
   :members:

.. automodule:: lumos.physics.coronagraphy
   :members:

.. automodule:: lumos.physics.fibers
   :members:

.. automodule:: lumos.physics.photonics
   :members:
```

## Elements

```{eval-rst}
.. automodule:: lumos.elements

.. automodule:: lumos.elements.sources
   :members:

.. automodule:: lumos.elements.apertures
   :members:

.. automodule:: lumos.elements.optics
   :members:

.. automodule:: lumos.elements.telescopes
   :members:

.. automodule:: lumos.elements.atmosphere
   :members:

.. automodule:: lumos.elements.coronagraphs
   :members:

.. automodule:: lumos.elements.fibers
   :members:

.. automodule:: lumos.elements.photonics
   :members:

.. automodule:: lumos.elements.detectors
   :members:
```

## Plotting

```{eval-rst}
.. automodule:: lumos.plotting
   :members:
```

```{eval-rst}
.. automodule:: lumos.plotting.style
   :members:

.. automodule:: lumos.plotting.results
   :members:
```

## Processing

```{eval-rst}
.. automodule:: lumos.processing.profiles
   :members:
```

## Results and archives

```{eval-rst}
.. automodule:: lumos.results.figures
   :members:

.. automodule:: lumos.results.archive
   :members: RunArchive, new_archive, default_archive_root, DEPENDENCIES

.. automodule:: lumos.results.reader
   :members:
```
