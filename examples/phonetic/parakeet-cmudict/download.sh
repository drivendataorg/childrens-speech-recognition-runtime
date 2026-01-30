#!/usr/bin/env bash

EXAMPLE_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

uvx hf download "nvidia/parakeet-tdt-0.6b-v2" \
    --local-dir "$EXAMPLE_ROOT_DIR/parakeet-tdt-0.6b-v2"
