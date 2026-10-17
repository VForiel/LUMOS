# Operating a bench

The same context runs in **simulation** or in **operation**. In operation, the actions of the elements (move
a stage, open a shutter, take an exposure) are sent to drivers, and the **digital twin** keeps simulating the
bench beside the equipment (ADR-0006, ADR-0014).

## Try it without equipment: the LumosLabs emulators

```text
lumos emulators list
lumos emulators start --port 5100
```

The second command serves the nine LumosLabs devices on `tcp://127.0.0.1:5100` to `5108` (local addresses
only, unless `--host` says otherwise). Their protocol is described in [](lumoslabs_protocol.md).

## Lab elements and drivers

{mod}`lumos.elements.lab` describes stages, piezo controllers, shutters, filter wheels, laser sources and phase
controllers; cameras and power meters are the detectors of {mod}`lumos.elements.detectors`. Attach a driver to
each element, by hand or by discovery:

```python
from lumos import Context
from lumos.elements.lab import LinearStage
from lumos.hardware.discovery import discover, associate

ctx = Context(mode="operation")
stage = ctx.add(LinearStage(name="stage"))
devices = discover(tcp=[f"127.0.0.1:{port}" for port in range(5100, 5109)])
associate(ctx, devices)  # attaches and connects LinearStageDriver to "stage"
stage.home()
stage.move(12 * u.mm)  # sent to the device, applied to the twin
```

`ctx.link("stage.position", "gap.distance", offset=180 * u.mm)` lets an actuator drive a parameter of the
optical model, so that the twin follows the bench.

## Safety

- **Limits**: every command is checked against the limits of its driver (read from the device), even when it
  is sent directly to the driver.
- **Dry run**: `driver.dry_run = True` checks and logs the commands without sending them.
- **Confirmations**: dangerous actions (switching a laser on) need `driver.confirm = callback`.
- **Emergency stop**: `ctx.stop_all()` stops every driver, from any thread, and brings the equipment to its
  safe state; drivers then refuse commands until `driver.reset()`.
- **Interruptions**: `with ctx.operation_guard(): ...` secures everything on Ctrl+C or on any error.
- **Recovery**: a lost connection is reopened and the command retried (`retries`, 2 by default); an
  equipment that does not answer raises `DriverTimeout`.

## Digital twin

In operation, `ctx.run(...)` returns the **measurements** of the detectors attached to drivers. With the twin
(the default), `result.twin` holds the simulated outputs and `result.residuals` their differences. Emulated
cameras and power meters receive the predictions of the twin and add their own noise and defects, so that the
residuals show a controlled gap; a fault injected in an emulator shows in the residuals.

## Sequences

```python
with ctx.record() as sequence:  # in simulation
    stage.home()
    stage.move(10 * u.mm)
sequence.save("alignment.json")

ctx_operation.replay(Sequence.load("alignment.json"))  # the same commands, sent to the equipment
```

The journal of a script is the same in simulation and in operation: only the measured durations differ.
