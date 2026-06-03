# Graph Test Fixtures

This directory contains small deterministic fixtures used by C++ graph tests.

- `gemma_forward_golden.bin`: 2 MB Gemma forward golden output fixture for
  regression tests. It is intentionally kept in the source tree because it is
  small, deterministic, and required to validate graph output compatibility.

Large pretrained weights, benchmark datasets, generated adapters, run logs, and
phone QA artifacts must stay outside this directory.
