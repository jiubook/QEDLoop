# Demo target for the self-iterating loop.
#
# Run the loop against it (offline, deterministic):
#     python run.py check --target examples/buggy_service
#     python run.py run   --target examples/buggy_service --provider mock
#
# The suite is RED on purpose: three defects are injected, each declared with a
# `# BUG: <name>` marker so the framework can measure whether a patch removed it.
