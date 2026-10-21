"""Documents from untrusted sources are refused cleanly and never execute anything (step P15.4, NF-7).

Requirements: a ``.lumos`` document (context, component, procedure) may come from a visitor of the public
instance or from the catalogue. Whatever it holds, reading and checking it raises only the documented error of
its reader (or returns a list of problems), and nothing in it is executed: no ``exec``/``eval``, no command,
no import of a module it names.

Property-based (Hypothesis): arbitrary JSON values, and documents of the right shape whose strings are
attempts at code.
"""

import builtins
import contextlib
import os
import subprocess
from unittest import mock

from hypothesis import given, settings
from hypothesis import strategies as st

from lumos.components import check_component
from lumos.core.procedure import Procedure, ProcedureError
from lumos.core.registry import UnknownTypeError
from lumos.io.lumos_format import LumosFormatError, from_document

#: The documented errors of ``from_document``: a malformed document, or an element type not registered.
REFUSED = (LumosFormatError, UnknownTypeError)

ATTACKS = [
    "__import__('os').system('echo pwned')",
    "${__import__('os')}",
    "$__class__",
    "1 m; import os",
    "eval('1')",
    "{{7*7}}",
    "../../../etc/passwd",
]

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=20)
    | st.sampled_from(ATTACKS),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=10), children, max_size=4)
    ),
    max_leaves=20,
)


@contextlib.contextmanager
def nothing_runs():
    """Fail if the code inside the block executes code, runs a command or starts a process."""

    def refuse(*args, **kwargs):
        raise AssertionError("an untrusted document tried to execute something")

    with (
        mock.patch.object(builtins, "exec", refuse),
        mock.patch.object(builtins, "eval", refuse),
        mock.patch.object(os, "system", refuse),
        mock.patch.object(subprocess, "Popen", refuse),
    ):
        yield


FUZZ = settings(max_examples=150, deadline=None)


@FUZZ
@given(component=json_values)
def test_any_component_document_gives_problems_not_errors(component):
    with nothing_runs():
        problems = check_component({"lumos_format": "1.3", "kind": "component", "component": component})
    assert isinstance(problems, list)
    assert all(isinstance(problem, str) for problem in problems)


@FUZZ
@given(
    template=st.sampled_from(["gain_phase", "spectral_transmission", "transfer_matrix", "composite", "x"]),
    body=st.dictionaries(
        st.sampled_from(["gain", "phase", "wavelengths", "transmission", "amplitudes", "phases", "group"]),
        json_values,
        max_size=4,
    ),
    params=st.dictionaries(st.sampled_from(["t", "__class__", "a b"]), json_values, max_size=2),
    checks=json_values,
)
def test_a_component_with_hostile_fields_gives_problems_not_errors(template, body, params, checks):
    component = {
        "id": "fuzz.Thing",
        "version": 1,
        "title": "Thing",
        "description": "fuzz",
        "license": "MIT",
        "authors": ["fuzz"],
        "template": template,
        "params": params,
        "body": body,
        "checks": checks,
    }
    with nothing_runs():
        problems = check_component({"lumos_format": "1.3", "kind": "component", "component": component})
    assert isinstance(problems, list)


@FUZZ
@given(data=json_values)
def test_any_procedure_is_read_or_refused(data):
    with nothing_runs():
        try:
            procedure = Procedure.from_json(data)
        except ProcedureError:
            return
        assert isinstance(procedure.problems(None), list)


@FUZZ
@given(
    steps=st.lists(
        st.fixed_dictionaries(
            {
                "do": st.sampled_from(
                    ["action", "measure", "process", "set", "use", "if", "for_each", "repeat"]
                )
            },
            optional={
                "element": json_values,
                "action": json_values,
                "function": json_values,
                "target": json_values,
                "value": json_values,
                "values": json_values,
                "times": json_values,
                "steps": json_values,
                "as": json_values,
            },
        ),
        max_size=4,
    )
)
def test_hostile_procedure_steps_are_reported_not_run(steps):
    from lumos import Context

    ctx = Context()
    with nothing_runs():
        try:
            procedure = Procedure.from_json({"name": "fuzz", "steps": steps})
        except ProcedureError:
            return
        assert isinstance(procedure.problems(ctx), list)


@FUZZ
@given(document=json_values)
def test_any_lumos_document_is_loaded_or_refused(document):
    with nothing_runs(), contextlib.suppress(*REFUSED):
        from_document(document)


@FUZZ
@given(
    kind=st.sampled_from(["context", "element", "group", "component"]),
    entries=st.lists(
        st.fixed_dictionaries(
            {"type": st.sampled_from(["os.system", "lumos.sources.GaussianLaser", "lumos.group", *ATTACKS])},
            optional={"type_version": json_values, "params": json_values, "name": json_values},
        ),
        max_size=3,
    ),
)
def test_documents_naming_modules_never_import_them(kind, entries):
    document = {
        "lumos_format": "1.3",
        "kind": kind,
        "elements": entries,
        "element": entries[0] if entries else {},
    }
    with nothing_runs(), contextlib.suppress(*REFUSED):
        from_document(document)
