#!/usr/bin/env bash
set -euo pipefail
# By default this produces parameter counts only.
# Add --train if you explicitly want to train randomly initialized controlled configs.
python experiments/parameter_sharing/run_parameter_sharing.py --continue_on_todo
