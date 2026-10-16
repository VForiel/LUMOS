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

.. automodule:: lumos.fields.spectral
   :members:

.. automodule:: lumos.fields.jones
   :members:
```

## Files

```{eval-rst}
.. automodule:: lumos.io
   :members:

.. automodule:: lumos.errors
   :members:
```
