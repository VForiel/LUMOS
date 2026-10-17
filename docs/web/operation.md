# Operating a bench

The same context that simulates a bench can drive the real equipment (operation mode, digital twin beside it;
see the hardware guide). The **Operation** tab of the interface appears for contexts with actions (stages,
shutters, lasers...).

## Without the operation enabled

Actions only change the simulation: press `home` or `move` (with arguments such as `5 mm`) to see what the
bench would do and how long it would take. Nothing real is driven, and no confirmation is asked.

## With the operation enabled

On your own machine, start the interface with:

```console
$ lumos web --enable-operation
```

(on a shared instance, see [Configuration](configuration.md): TLS and registered operators are required).

1. Create or open a context in operation mode.
2. In the **Operation** tab, give the addresses of the devices (`tcp://192.168.1.20:5025`, or the LumosLabs
   emulators started with `lumos emulators start`) and press **Discover**: LUMOS asks each address who it is.
3. Press **Associate** to attach the devices found to the elements of the context.
4. Run the actions: each action on real equipment opens a dialog naming the device, and runs only once you
   confirm it. Limits are checked by the server, whatever the request.
5. The **STOP** button of the toolbar sends an emergency stop to every device of the context, at any time and
   whatever your role.

Every action is written to the journal of the context (origin `web`) and, on a shared instance, to the audit
journal. Only operators can discover devices and act on equipment; simulators and viewers see their state.
