#!/usr/bin/env python3
"""R06 resource matrix: 16384 FFMA per lane (8 chains x 2048) for all six cases."""
from run_r01 import main


def matrix():
    return [dict(id=f'ffma_r{r}_{scope}', registers=r, scope=scope, steps=2048)
            for r in (64, 96, 128) for scope in ('one_cta', 'all_gpu')]


if __name__ == '__main__':
    main('r06', matrix())
