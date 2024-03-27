import pytest
import torch

from fms_extras.models.speculator import apply_index_map, flatten_batch


def _get_test_inp():
    return list(torch.randn(100, 4, 20, 4).sign())


def test_flatten_correctness():
    # Verify that compression is occurring, and flattened batch corresponds to the flattening map
    inps = _get_test_inp()
    for inp in inps:
        inp_flat, _, ind_map = flatten_batch(inp)
        # There are only 8 possible unique candidates, from a set of 20, so compression is at least 2x
        assert inp_flat.numel() < inp.numel() // 2
        torch.testing.assert_close(inp_flat, apply_index_map(inp.view(-1), ind_map))


def test_flatten_unflatten():
    # Verify that unflat(flat(x)) == x
    inps = _get_test_inp()
    for inp in inps:
        _, unflat_map, flat_map = flatten_batch(inp)
        new_inp = apply_index_map(apply_index_map(inp.view(-1), flat_map), unflat_map)
        torch.testing.assert_close(inp, new_inp)


def test_unflatten_flatten():
    # Verify that flat(unflat(x)) == x
    inps = _get_test_inp()
    for inp in inps:
        inp_flat, unflat_map, flat_map = flatten_batch(inp)
        new_inp = apply_index_map(
            apply_index_map(inp_flat, unflat_map).view(-1), flat_map
        )
        torch.testing.assert_close(inp_flat, new_inp)
