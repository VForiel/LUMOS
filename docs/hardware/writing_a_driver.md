# Writing a driver

A driver connects an element of LUMOS to the equipment it represents. Writing one means subclassing
{class}`lumos.core.driver.Driver` (or {class}`lumos.hardware.drivers.lumoslabs.LumosLabsDriver` for a device
speaking lines of ASCII), then registering it for the models it supports. The safety mechanisms (limits,
dry run, confirmations, recovery, emergency stop, safe state) come from the base class: a driver only
implements how to talk to its device.

## The interface

| Method or attribute | What to implement |
| --- | --- |
| `_connect()` | open the connection (SDK handle, transport), check the identity of the device, read its limits |
| `_disconnect()` | close it |
| `_send(command)` | execute a {class}`~lumos.core.driver.Command` (`command.action`, `command.args`, `command.kwargs`) and return a {class}`~lumos.core.driver.Reply` with the state read back (`state={"position": ...}`) |
| `_query(name, *arguments)` | read a quantity (`"position"`, `"power"`...) and return it as a `Quantity` |
| `_stop()` | interrupt any motion or emission **at once**; it is called from another thread while `_send` may be waiting, so it must not block on the same lock |
| `_safe_state()` | bring the device to its safe state (laser off, shutter closed, heaters off) |
| `signatures` | names of the positional arguments of each action, e.g. `{"move": ("target",)}` |
| `limits` | software limits `{"move.target": (0 * u.mm, 25 * u.mm)}`, completed at connection from the device |
| `dangerous` / `is_dangerous(command)` | actions that need the confirmation callback (emission of a laser...) |

Inside `_send`, wait for long operations with `self.wait(seconds)`, which returns `False` as soon as an
emergency stop occurs; then raise {class}`~lumos.core.driver.DriverStopped`. Report a lost connection with
{class}`~lumos.core.driver.ConnectionLost` (the base class reconnects and retries) and an equipment that does
not answer with {class}`~lumos.core.driver.DriverTimeout`.

Measuring devices also provide `configure(...)`, `acquire()`, which returns a
{class}`~lumos.fields.DataProduct` like the simulated detector (REQ-HW-5), and optionally `feed(...)` for
emulated devices fed by the digital twin.

## A minimal driver

```python
from astropy import units as u
from lumos.core.driver import Driver, DriverStopped, Reply


class MyStageDriver(Driver):
    signatures = {"move": ("target",), "home": ()}

    def __init__(self, serial, **options):
        super().__init__(**options)
        self.serial = serial

    def _connect(self):
        self.handle = vendor_sdk.open(self.serial)
        low, high = vendor_sdk.travel(self.handle)
        self.limits["move.target"] = (low * u.mm, high * u.mm)

    def _disconnect(self):
        vendor_sdk.close(self.handle)

    def _send(self, command):
        if command.action == "move":
            vendor_sdk.move_to(self.handle, command.args[0].to_value(u.mm))
            while vendor_sdk.is_moving(self.handle):
                if not self.wait(0.02):
                    raise DriverStopped("move interrupted")
            return Reply(state={"position": self._query("position")})
        ...

    def _query(self, name, *arguments):
        return vendor_sdk.position(self.handle) * u.mm

    def _stop(self):
        vendor_sdk.stop_immediate(self.handle)
```

## Registering it

In the `pyproject.toml` of your package, declare one entry point per model in the group `lumos.drivers`:

```toml
[project.entry-points."lumos.drivers"]
KDC101 = "lumos_thorlabs.kinesis:KCubeDCServoDriver"
PM100D = "lumos_thorlabs.tlpm:PowerMeterDriver"
```

{func}`lumos.hardware.drivers.driver_for` then finds them, and the discovery associates them with the elements
whose `device_model` (or `lumoslabs_model`) matches. In a script, `register_driver(model, cls)` does the same.

## Testing it

Test the driver against an emulator or a fake transport, like the LumosLabs drivers: limits refused even for
commands sent directly, an emergency stop interrupting a long operation, recovery after a lost connection,
and a measured duration consistent with the duration predicted by the element. The LumosLabs emulators
(`lumos emulators start`) are a reference implementation of these behaviours.

## Coverage of the Thorlabs SDKs

The first vendor drivers will target Thorlabs. Their SDKs map onto the interface as follows (from the public
documentation of the SDKs; to be confirmed on the hardware):

| SDK | Operation | Interface of LUMOS |
| --- | --- | --- |
| Kinesis (motion, e.g. `CC_*` of the KCube DC servo) | `CC_Open`, `CC_Close`, `CC_StartPolling` | `_connect`, `_disconnect` |
| Kinesis | `CC_Home`, `CC_MoveToPosition`, status bits | `_send` (`home`, `move`) waiting with `self.wait` |
| Kinesis | `CC_GetPosition`, device units ↔ real units | `_query("position")`, quantities |
| Kinesis | `CC_SetVelParams` | velocity and acceleration of the element, as for `LL-LS100` |
| Kinesis | `CC_StopImmediate` / `CC_StopProfiled` | `_stop` / `_safe_state` |
| Kinesis | travel limits of the stage | `limits` read in `_connect` |
| Kinesis | `TLI_BuildDeviceList`, `TLI_GetDeviceListByTypeExt` | discovery: **gap**, the discovery identifies devices by `*IDN?` over transports; SDK enumeration needs a discovery hook for plugins |
| TSI cameras (`thorlabs_tsi_sdk`) | `discover_available_cameras`, `open_camera` | `_connect` (enumeration: same gap) |
| TSI | `exposure_time_us`, gain, black level, `roi` | `configure(exposure=..., gain=..., roi=...)` |
| TSI | `arm`, `issue_software_trigger`, `get_pending_frame_or_null`, `disarm` | `acquire()` returning a `DataProduct` in adu, polling with `self.wait` |
| TSI | hardware triggering, continuous streaming | **gap**: the camera element has no trigger mode nor frame stream yet |
| TLPM (power meters, VISA) | `TLPM_findRsrc`, `TLPM_init`, `TLPM_close` | discovery over VISA (`*IDN?` answers `Thorlabs,PM100D,...`), `_connect` |
| TLPM | `TLPM_setWavelength`, `TLPM_setPowerRange`, `TLPM_setAvgTime` | `configure(wavelength=..., range=..., integration=...)` |
| TLPM | `TLPM_measPower` | `acquire()` returning a `DataProduct` in W |

The two gaps (enumeration through vendor SDKs, hardware triggering and streaming of cameras) are to be closed
with the first Thorlabs drivers.
