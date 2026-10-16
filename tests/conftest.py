"""Shared pytest configuration."""

import os

from hypothesis import settings

# "dev" keeps local runs fast; CI sets HYPOTHESIS_PROFILE=ci for a deeper search.
settings.register_profile("dev", max_examples=50)
settings.register_profile("ci", max_examples=300, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
