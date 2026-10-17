# LumosLabs protocol

LumosLabs is a fictitious range of laboratory equipment. Its devices do not exist, but their **emulators** speak
the protocol below over real transports (TCP, serial), so that LUMOS drives them exactly as it drives physical
equipment: same drivers, same errors, same latencies (ADR-0014). The protocol is also a template for vendors.

## Lines

- One command per line, one reply per command, lines terminated by `\n` (ASCII).
- A command is `VERB [argument ...]`. Verbs are case-insensitive; queries end with `?`.
- Tokens are separated by spaces:

| Token | Example | Meaning |
| --- | --- | --- |
| integer | `3` | channel, slot, count |
| real | `0.25`, `1e-12` | dimensionless value |
| quantity | `12.5[mm]`, `4[mm/s]`, `1.5[V]` | value and astropy unit, without space |
| word | `ON`, `X`, `READY` | keyword |
| array | `float32 3x4 b64:...` | dtype, shape, base64 of the little-endian bytes |

- A reply is `OK [value ...]` or `ERR <code> <message>`.

## Error codes

| Code | Name | Raised by the driver as |
| --- | --- | --- |
| `100` | unknown command | `DeviceError` |
| `101` | bad argument (missing, wrong type) | `DeviceError` |
| `102` | bad unit (not convertible) | `DeviceError` |
| `200` | out of range | `DeviceLimitError` (a `LimitError`) |
| `300` | busy (another command is in progress) | `DeviceBusy` |
| `301` | not homed (a move needs `HOME` first) | `DeviceError` |
| `400` | emergency stop latched (cleared by `*RST`) | `DeviceStopped` (a `DriverStopped`) |
| `500` | hardware fault | `DeviceError` |
| `501` | motion stalled | `DeviceError` |

## Common commands

| Command | Reply | Effect |
| --- | --- | --- |
| `*IDN?` | `OK LUMOSLABS,<model>,<serial>,<firmware>` | identification |
| `*RST` | `OK` | stops everything, clears the emergency stop and the errors |
| `STATUS?` | `OK READY` / `BUSY` / `STOPPED` / `FAULT` | state of the device |
| `BUSY?` | `OK 0` / `OK 1` | whether a long operation (move, exposure) is in progress |
| `STOP` | `OK` | emergency stop: interrupts the operation and latches `STOPPED` |
| `LIMITS? <name>` | `OK <low> <high>` | range of an adjustable quantity |

Long operations (moves, slot changes, exposures, warm-up) return `OK` at once; the driver polls `BUSY?` until
`OK 0`, which lets `STOP` interrupt them.

## Devices

### `LL-LS100` linear stage (100 mm)

| Command | Reply |
| --- | --- |
| `HOME` | `OK` (long operation, finds the reference switch) |
| `MOVE <position[mm]>` | `OK` (long, `ERR 301` before `HOME`, `ERR 200` outside `LIMITS? POS`) |
| `POS?` | `OK <position[mm]>` (encoder reading, with a small position error) |
| `VEL <speed[mm/s]>`, `VEL?` | velocity (trapezoidal profile with the acceleration `ACC`) |
| `ACC <acceleration[mm/s2]>`, `ACC?` | acceleration |

### `LL-RS360` rotation stage

Same commands in degrees (`MOVE 90[deg]`, `VEL 20[deg/s]`); `LIMITS? POS` is ±360 deg.

### `LL-PZ3` three-axis piezo controller

| Command | Reply |
| --- | --- |
| `MOVE <axis> <position[um]>` | `OK`, axis `X`, `Y` or `Z`, range 0–30 µm (`LIMITS? POS`) |
| `POS? <axis>` | `OK <position[um]>` (with hysteresis and noise) |

### `LL-SH1` shutter

`OPEN`, `CLOSE` (long, 8 ms and 10 ms) and `STATE?` → `OK OPEN` / `CLOSED` / `MOVING`.

### `LL-FW6` filter wheel

`SLOT <n>` (1–6, long: 0.4 s per step, shortest direction) and `SLOT?` → `OK <n>`.

### `LL-LD` laser source

| Command | Reply |
| --- | --- |
| `EMIT ON` / `EMIT OFF` | `OK` (emission; the driver asks a confirmation before `ON`) |
| `EMIT?` | `OK ON` / `OFF` |
| `POWER <power[mW]>`, `POWER?` | set point; `POWER?` returns the emitted power, with warm-up and fluctuations |
| `WL?` | `OK <wavelength[nm]>` |

### `LL-PM` power meter

| Command | Reply |
| --- | --- |
| `WL <wavelength[nm]>` | correction wavelength |
| `INT <time[s]>`, `INT?` | integration time |
| `RANGE <power[W]>`, `RANGE?` | full scale (readings above it saturate) |
| `READ?` | `OK <power[W]>`: the incident power, with offset, gain error and noise |

### `LL-CAM` scientific camera

| Command | Reply |
| --- | --- |
| `EXP <time[s]>`, `EXP?` | exposure time |
| `GAIN <electrons per adu>`, `GAIN?` | conversion gain |
| `ROI <x> <y> <width> <height>`, `ROI?` | region of interest |
| `ACQ` | `OK` (long: the exposure) |
| `FRAME?` | `OK uint16 <h>x<w> b64:...`: the last image in adu |

### `LL-PCU` phase controller of photonic chips

| Command | Reply |
| --- | --- |
| `CHAN?` | `OK <n>` number of heater channels |
| `VOLT <channel> <voltage[V]>`, `VOLT? <channel>` | heater voltage, 0–10 V, settling time 5 ms |
| `PHASE? <channel>` | `OK <phase[rad]>`: phase produced, `κV²` plus thermal crosstalk from the neighbours |

## Simulation commands

The emulators of the devices that measure light accept data from the digital twin; real devices ignore them
(`ERR 100`).

| Command | Effect |
| --- | --- |
| `SIM:POWER <power[W]>` | incident power seen by `LL-PM` |
| `SIM:IMAGE <dtype> <h>x<w> b64:...` | expected image of `LL-CAM`, in electrons per second |
| `SIM:FAULT <kind> [value]` | injects a fault: `DISCONNECT`, `DELAY <time[s]>`, `OFFSET <value>`, `STUCK`, `CLEAR` |

The emulators add their own defects to what the twin sends them (noise, offset, gain error, hot pixels): the
measured and simulated outputs then differ in a controlled way.
