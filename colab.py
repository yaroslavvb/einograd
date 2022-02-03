"""Base types used everywhere"""
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from typing import Union

import more_itertools
import natsort
import numpy as np
import opt_einsum as oe
import torch
from opt_einsum import helpers as oe_helpers
from torch import nn as nn

import math
import sys

import pytest
import torch



def freeze_multimap(idx_to_dim) -> Dict[Any, Tuple]:
    """Freezes dictionary {a->[], b->[]}"""
    # TODO(y) doesn't fully freeze since dictionaries are mutable
    d = {}
    for (key, value) in idx_to_dim.items():
        assert isinstance(value, List) or isinstance(value, Tuple)
        d[key] = tuple(value)
    return d


# GLOBALS = AttrDict({'DEBUG': True, 'device': 'cpu', 'PURE_TENSOR_NETWORKS': False,
#               'tensor_count': 0, 'ALLOW_PARTIAL_CONTRACTIONS': False,
#               'ALLOW_UNSORTED_INDICES': False,
#               'MAX_INDEX_COUNT': 1000, 'idx0': 'a'})

class _GLOBALS_CLASS:
    __shared_state = {}

    idx0: str
    function_count: Dict[str, int]
    function_dict: Dict[str, 'Function']
    tensor_count: int
    global_forward_flops: int

    def __init__(self):
        self.init_values()

    def reset_global_state(self):
        self.init_values()

    def init_values(self):
        self.FULL_HESSIAN = True
        self.switch_composition_order = False
        self.global_forward_flops = 0
        self.device = 'cpu'
        self.PURE_TENSOR_NETWORKS = False
        self.tensor_count = 0
        self.function_count = {}  # {LinearLayer: 1, Relu: 2}
        self.function_dict = {}  # {LinearLayer: 1, Relu: 2}
        self.ALLOW_PARTIAL_CONTRACTIONS = True  # allow some indices of the left tensor to remain uncontracted
        self.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False  # the original order turned out to be wrong for the Hessian case
        self.ALLOW_UNSORTED_INDICES = False
        self.MAX_INDEX_COUNT = 1000
        self.idx0 = 'a'
        self.all_indices = set(chr(ord(self.idx0) + i) for i in range(self.MAX_INDEX_COUNT))
        self.all_indices_list = tuple(chr(ord(self.idx0) + i) for i in range(self.MAX_INDEX_COUNT))
        self.enable_memoization = False
        self.debug_print = False

    def p(self, *args):
        if self.debug_print:
            print(*args)

    def generate_tensor_name(self):
        name = f'T{self.tensor_count:02d}'
        self.tensor_count += 1
        return name

    def reset_function_count(self):
        self.function_count = {}
        self.function_dict = {}

    def reset_tensor_count(self):
        self.tensor_count = 0

    def increment_global_forward_flops(self, incr):
        self.global_forward_flops += incr

    def reset_global_forward_flops(self):
        self.global_forward_flops = 0

    def get_global_forward_flops(self):
        return self.global_forward_flops


GLOBALS = _GLOBALS_CLASS()


##################################################
# Tensors
##################################################


class TensorSharedImpl(ABC):
    # addition
    def __add__(self, other: 'Tensor'):
        assert isinstance(other, Tensor)
        if isinstance(other, ZeroTensor):
            return self
        if isinstance(self, TensorAddition):
            return TensorAddition(self.children + [other])
        else:
            return TensorAddition([self, other])


class Tensor(ABC):
    """
    Tensor object, corresponding to multilinear function, is characterized by a set of indices
    and their dimensions.

    Additionally, indices are split into "input" and "output" indices.

    For multilinear functions (like Hessian and higher derivatives), input indices are partitioned
    corresponding to multiple inputs. IE,
    ie, {out:ab in:{cd,ef}}

    Consider specifying symmetric structure explicitly
    ie, {out:ab, in:{cd: 2}}

    """

    @property
    def flops(self):
        return -492

    @property
    def value(self):
        return torch.Tensor(-498)


#    
#    def in_dims(self):
#        pass

#    
#    def out_dims(self):
#        pass


class AtomicTensor(Tensor, ABC):
    pass


class TensorAddition(TensorSharedImpl):
    def __init__(self, children: List['Tensor']):
        # Must have two children. Otherwise, it's harder to tell if two functions are the same
        # ie, FunctionAddition([f]) == FunctionContraction([f])
        assert len(children) >= 2
        self.children = children

    @property
    def value(self):
        result = self.children[0].value
        for c in self.children[1:]:
            result = result + c.value
        return result

    @property
    def flops(self):
        total_flops = 0
        for child in self.children:
            total_flops += child.flops
        return total_flops


# TODO(y): move order to (tensor, idx, ...)
class TensorContraction(Tensor, TensorSharedImpl):
    label: str  # tag helpful for debugging

    # see UnitTestB for illustration
    _original_specs: Tuple[Tuple]  # [('ab|cd', A, 'A'), ...]
    children_specs: List[str]  # [ab|cd, c|c, cd|ef], has to be a list because of relabeling function
    children_data: Tuple[torch.Tensor]  # [ones((2,3,2,2)), ones((2,), ones((2, 2, 2, 4))]
    children_labels: Tuple[str]  # ['A', 'B', 'C']

    out_idx: List[chr]  # [a, b]
    contracted_idx: List[chr]  # [c, d]
    diag_idx: List[chr]  # ie, i|i kind of tensor
    in_idx: List[chr]  # [e, f]

    idx_to_out_tensors: Dict[chr, Tuple[int]]  # tensors for which this index is contravariant
    idx_to_in_tensors: Dict[chr, Tuple[int]]  # tensors for which this index is covariant
    idx_to_diag_tensors: Dict[chr, Tuple[int]]  # tensors for which this index is both (co/contra)variant

    idx_to_dim: Dict[chr, int]  # {a:2, b:3, c:2, d:2, e:2, f:4}

    _einsum_str: str  # 'abcd,c,cdef->abef'
    ricci_str: str  # ab|cd,cd|ef->ab|ef, Like einsum string, but uses Ricci calculus distinction of up/down indices, separating them by |
    transposed: bool

    @property
    def ricci_in(self):
        return self.ricci_str.split('->')[0]

    @property
    def ricci_out(self):
        return self.ricci_str.split('->')[1]

    @staticmethod
    def __legacy_init__(index_spec_list, tensors, label=None) -> 'TensorContraction':
        return TensorContraction(list((i, t) for (i, t) in zip(index_spec_list, tensors)), label=label)

    def print_full_spec(self):
        for spec, data, label in zip(self.children_specs, self.children_data, self.children_labels):
            print(label)
            print(spec)
            print(data)
            print('------')

    def __init__(self, specs: Union[List[Tuple], Tuple[Tuple]], copy_tensors=None, label=None):
        """

        Args:
            specs: list of tensor specs, tuples like ('ab|cd', A, 'A'). 'ab' are output/contravariant indices
            label: optional label of the Tensor, used in printing

        Main job of this function is to figure out which indices get contracted and which ones are left over.
        For the left-over ones, determine whether they are input, output or both.

        For instance a_i b_i^i -> results in _i being an input index
        """
        self._original_specs = tuple(specs)
        self._einsum_spec = None
        self.copy_tensors = copy_tensors

        children_specs: List[str] = []
        tensors: List[torch.Tensor] = []
        tensor_labels: List[str] = []
        for spec in specs:
            if len(spec) == 3:
                (tensor_spec, tensor_data, tensor_label) = spec
                assert isinstance(tensor_label, str), f"Must provide string label for tensor, instead provided type {type(tensor_label)}"
            else:
                assert len(spec) == 2, f"Got spec {spec}, with {len(spec)} terms, expected form i|j,A,'A' or i|j,A"
                (tensor_spec, tensor_data) = spec
                tensor_label = GLOBALS.generate_tensor_name()
                assert isinstance(tensor_label, str), "internal error"

            assert isinstance(tensor_spec, str)
            assert isinstance(tensor_data, torch.Tensor), f"Provided not an instance of torch.Tensor for {tensor_spec}, " \
                                                          f"{tensor_label}, instead see type {type(tensor_data)}"

            children_specs.append(tensor_spec)
            tensors.append(tensor_data)
            tensor_labels.append(tensor_label)

        self.children_specs = list(children_specs)
        self.children_data = tuple(tensors)
        self.children_labels = tuple(tensor_labels)
        self.label = label if label is not None else GLOBALS.generate_tensor_name()

        # see "Diagonal logic" notes
        # map tensors to dimensions
        idx_to_dim: Dict[chr, int] = {}
        idx_to_out_tensors: Dict[chr, List[int]] = {}
        idx_to_in_tensors: Dict[chr, List[int]] = {}
        idx_to_diag_tensors: Dict[chr, List[int]] = {}
        # maybe use list instead of set to maintain order?
        all_indices = set(''.join(children_specs).replace('|', ''))
        assert ' ' not in all_indices
        for idx in all_indices:
            idx_to_out_tensors.setdefault(idx, [])
            idx_to_in_tensors.setdefault(idx, [])
            idx_to_diag_tensors.setdefault(idx, [])

        # for each index, get a list of tensors for which it's contravariant-only (out), covariant-only (in) or both (diagonal)
        for tensor_id, (tensor_spec, tensor_data) in enumerate(zip(self.children_specs, self.children_data)):
            out_idx_term, in_idx_term = tensor_spec.split('|')
            for idx in out_idx_term:
                if idx not in in_idx_term:
                    idx_to_out_tensors[idx].append(tensor_id)
            for idx in in_idx_term:
                if idx not in out_idx_term:
                    idx_to_in_tensors[idx].append(tensor_id)
            for idx in set(out_idx_term + in_idx_term):
                if idx in out_idx_term and idx in in_idx_term:
                    idx_to_diag_tensors[idx].append(tensor_id)

            # some logic assumes indices are alphabetically sorted and the original order is lost (because of using set() rather than
            # list() to manipulate indices). Ensure indices are alphabetically sorted, this means order is not lost
            assert more_itertools.is_sorted(out_idx_term, strict=True)
            assert more_itertools.is_sorted(in_idx_term, strict=True)

            # get index shape from provided tensor
            assert len(tensor_data.shape) == len(set(out_idx_term + in_idx_term)), f"Not enough dimensions for indices, " \
                                                                                   f"have {len(out_idx_term + in_idx_term)} indices, {len(set(out_idx_term + in_idx_term))} unique but " \
                                                                                   f"{len(tensor_data.shape)} dimensions"
            for (idx, dim) in zip(out_idx_term + in_idx_term, tensor_data.shape):
                if idx in idx_to_dim:
                    assert idx_to_dim[idx] == dim, f"trying to set idx {idx} in {tensor_spec} to {dim}, " \
                                                   f"but it's already set to have dimension {idx_to_dim[idx]}"
                assert dim > 0, f"Index {idx} in {tensor_spec} must have positive dimension, instead see {dim}"
                idx_to_dim[idx] = dim

        self.idx_to_dim = idx_to_dim  # TODO(y): use frozendict
        self.idx_to_out_tensors = freeze_multimap(idx_to_out_tensors)
        self.idx_to_in_tensors = freeze_multimap(idx_to_in_tensors)
        self.idx_to_diag_tensors = freeze_multimap(idx_to_diag_tensors)

        out_idx = []
        in_idx_order = []  # list of in indices along with tensor. Use this to sort indices of later tensor ahead of former
        in_idx = []
        contracted_idx = []
        # # should be |j   but have j|:  |i * i|i. i is "is_in" and "is_diag"

        for idx in self.idx_to_dim.keys():
            is_out = len(self.idx_to_out_tensors[idx]) > 0
            is_in = len(self.idx_to_in_tensors[idx]) > 0
            is_diag = len(self.idx_to_diag_tensors[idx]) > 0
            if is_out and is_in:
                contracted_idx.append(idx)
            elif is_out and is_diag:  # contracted with diagonal on left
                out_idx.append(idx)
                in_idx_order.append((idx, self.idx_to_diag_tensors[idx]))
            elif is_in and is_diag:  # contracted with diagonal on right
                in_idx_order.append((idx, self.idx_to_diag_tensors[idx]))
                in_idx.append(idx)
            elif is_diag:  # passthrough without multiplication
                in_idx_order.append((idx, self.idx_to_diag_tensors[idx]))
                out_idx.append(idx)
                in_idx.append(idx)
            elif is_out:  # passhtrough to left without contraction
                out_idx.append(idx)
            elif is_in:  # passthrough to right without contraction
                in_idx_order.append((idx, self.idx_to_in_tensors[idx]))
                in_idx.append(idx)
            else:
                assert False, "index is neither out, in or diagonal, how did this happen?"

        # Arrange output indices to order right-most tensors output indices first (see UnitTestC)
        new_in_idx = []
        GLOBALS.p('========')
        GLOBALS.p(in_idx_order)
        # for each in index determine the largest index tensor for which it's an in-index
        in_idx_to_rightmost_tensor = {}
        for (idx, tensor_id_tuple) in in_idx_order:
            for tensor_id in tensor_id_tuple:
                in_idx_to_rightmost_tensor[idx] = max(in_idx_to_rightmost_tensor.get(idx, -1), tensor_id)

        GLOBALS.p('in_idx_order ', in_idx_order)
        # GLOBALS.p('index i has rank', in_idx_to_rightmost_tensor['i'])
        for rightmost_tensor_id in reversed(sorted(in_idx_to_rightmost_tensor.values())):
            for (idx, tensor_id_tuple) in in_idx_order:
                if max(tensor_id_tuple) != rightmost_tensor_id:
                    continue
                if idx in in_idx and not idx in new_in_idx:
                    new_in_idx.append(idx)

        in_idx = new_in_idx
        self.out_idx = out_idx
        self.contracted_idx = contracted_idx
        self.in_idx = in_idx
        self.diag_idx = [x for x in out_idx if x in in_idx]  # intersect while preserving order

        # delete unique index from list
        def del_idx(ll, idx_name):
            assert ll.count(idx_name) == 1, f"Deleting non-existent index {idx_name} from {ll}"
            del ll[ll.index(idx_name)]

        # special handling for two kinds of copy tensors, diagonal operator and trace
        if copy_tensors is not None:
            assert isinstance(copy_tensors, List)

            assert len(copy_tensors) == 1
            if len(copy_tensors[0]) == 4:  # ij|k case
                # 1. both indices must be in-indices
                # 2. rename top index to match bottom
                # 3. update input indices
                in1, in2, _, out1 = copy_tensors[0]
                assert in1 in self.in_idx
                assert in2 in self.in_idx
                assert in1 != in2
                assert self.idx_to_dim[in1] == self.idx_to_dim[in2]
                del_idx(self.in_idx, in2)
                self._rename_index(in2, in1, allow_diagonal_rewiring=True)

                einsum_in = ','.join(self._process_for_einsum(tensor_spec, allow_repetitions=True) for tensor_spec in self.children_specs)
                einsum_out = ''.join(self.out_idx) + ''.join(self.in_idx)
                self._einsum_spec = f'{einsum_in}->{einsum_out}'
                self.ricci_str = f"{','.join(self.children_specs)}->{''.join(list(self.out_idx))}|{''.join(list(self.in_idx))}"
            elif len(copy_tensors[0]) == 3:  # ii| case
                in1, in2, _, = copy_tensors[0]
                assert in1 in self.in_idx
                assert in2 in self.in_idx
                assert in1 != in2
                assert self.idx_to_dim[in1] == self.idx_to_dim[in2]
                del_idx(self.in_idx, in1)
                del_idx(self.in_idx, in2)
                self._rename_index(in2, in1, allow_diagonal_rewiring=True)

                einsum_in = ','.join(self._process_for_einsum(tensor_spec, allow_repetitions=True) for tensor_spec in self.children_specs)
                einsum_out = ''.join(self.out_idx) + ''.join(self.in_idx)
                self._einsum_spec = f'{einsum_in}->{einsum_out}'
                self.ricci_str = f"{','.join(self.children_specs)}->{''.join(list(self.out_idx))}|{''.join(list(self.in_idx))}"
            else:
                assert False, f"Received copy tensors {copy_tensors}, haven't tested this case yet"
        else:
            # assert self.label != 'T60'
            self.ricci_str = f"{','.join(self.children_specs)}->{''.join(list(self.out_idx))}|{''.join(list(self.in_idx))}"

            if not self.diag_idx:  # einsum can't materialize diagonal tensors, don't generate string here
                einsum_in = ','.join(self._process_for_einsum(tensor_spec) for tensor_spec in self.children_specs)
                einsum_out = ''.join(self.out_idx) + ''.join(self.in_idx)
                self._einsum_spec = f'{einsum_in}->{einsum_out}'
            else:
                GLOBALS.p("diagonal tensor, no einsum for " + ','.join(self.children_specs))
                self._einsum_spec = None  # unsupported by torch.einsum

    @staticmethod
    def _process_for_einsum(spec, allow_repetitions=False):
        """Processes tensor spec for einsum. Drop upper/lower convention, dedup indices occurring both as upper/lower,
        which happens for diagonal tensors"""
        out_spec, in_spec = spec.split('|')
        if not allow_repetitions:
            assert len(out_spec) == len(set(out_spec)), f"Index occurs multiple times as output in {spec}"
            assert len(in_spec) == len(set(in_spec)), f"Index occurs multiple times as output in {spec}"

        new_in_spec = []
        for c in in_spec:
            if c in out_spec:
                continue
            new_in_spec.append(c)
        return out_spec + ''.join(new_in_spec)

    @property
    def out_dims(self):
        return tuple(self.idx_to_dim[c] for c in self.out_idx)

    @property
    def in_dims(self):
        return tuple(self.idx_to_dim[c] for c in self.in_idx)

    @property
    def flops(self):
        """Flops required to materialize this tensor after einsum optimization"""

        views = oe.helpers.build_views(self._einsum_spec, self.idx_to_dim)
        path, info = oe.contract_path(self._einsum_spec, *views, optimize='dp')
        return int(info.opt_cost)

    def _print_schedule(self):
        """Prints contraction schedule obtained by einsum optimizer"""

        einsum_str = self._einsum_spec
        sizes_dict = self.idx_to_dim

        # indices: ['ij','jk','kl','lm']
        indices = einsum_str.split('->')[0].split(',')
        output_indices = einsum_str.split('->')[1]
        # unique_inds = set(einsum_str) - {',', '-', '>'}
        # index_size = [5]*len(unique_inds)
        # sizes_dict = dict(zip(unique_inds, index_size))
        views = oe.helpers.build_views(einsum_str, sizes_dict)

        # path: contraction path in einsum optimizer format, ie, [(0,), (2,), (1, 3), (0, 2), (0, 1)]
        path, info = oe.contract_path(einsum_str, *views, optimize='dp')

        # TODO(y): replace terms with something user provided
        # terms: ['term1', 'term2', 'term3', 'term4']
        terms = [f'term{i}' for i in range(len(indices))]
        GLOBALS.p('optimizing ', einsum_str, terms)
        GLOBALS.p('flops: ', info.opt_cost)

        # output_subscript: ['kl']
        output_subscript = output_indices

        input_index_sets = [set(x) for x in indices]
        output_indices = frozenset(output_subscript)

        derived_count = 0
        for i, contract_inds in enumerate(path):
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            # GLOBALS.p(f'contracting {contract_inds}, input {input_index_sets}, output {output_indices}')
            contract_tuple = oe_helpers.find_contraction(contract_inds, input_index_sets, output_indices)
            out_inds, input_index_sets, _, idx_contract = contract_tuple
            # GLOBALS.p(f'idx_contract {idx_contract}, out_inds {out_inds}')

            current_input_index_sets = [indices.pop(x) for x in contract_inds]
            current_terms = [terms.pop(x) for x in contract_inds]

            # Last contraction
            if (i - len(path)) == -1:
                current_output_indices = output_subscript
                derived_term = f'derived{derived_count}'
            else:
                all_input_inds = "".join(current_input_index_sets)
                current_output_indices = "".join(sorted(out_inds, key=all_input_inds.find))
                derived_term = f'derived{derived_count}'
                derived_count += 1

            indices.append(current_output_indices)
            terms.append(derived_term)

            new_terms = []
            new_sets = []
            # for i in natsort.index_natsorted(current_terms):
            for i in natsort.index_natsorted(current_input_index_sets):
                new_terms.append(current_terms[i])
                new_sets.append(current_input_index_sets[i])
            # einsum_str = ",".join(current_input_index_sets) + "->" + current_output_indices
            #        GLOBALS.p(f'{derived_term}=einsum({einsum_str}, {current_terms})')
            einsum_str = ",".join(new_sets) + "->" + current_output_indices
            GLOBALS.p(f'{derived_term}=einsum({einsum_str}, {new_terms})')

    def _check_indices_sorted(self):
        assert more_itertools.is_sorted(self.out_idx, strict=True)
        assert more_itertools.is_sorted(self.contracted_idx, strict=True)
        # don't check if input indices are sorted because it fails for partial contractions. IE
        # ab|cd * c|d => ab|ed, input indices no longer sorted for partial contractions

    def __str__(self):
        out_dim_spec = ','.join(str(d) for d in self.out_dims)
        in_dim_spec = ','.join(str(d) for d in self.in_dims)
        dim_spec = f"{out_dim_spec}|{in_dim_spec}"
        return f"{dim_spec} '{self.label}', out({','.join(self.out_idx)}), in({','.join(self.in_idx)}), spec: {','.join(self.children_specs)}"

    def __repr__(self):
        return self.__str__()

    def _rename_index(self, old_name: str, new_name: str, allow_diagonal_rewiring=False):
        """

        Args:
            old_name:
            new_name:
            allow_diagonal_rewiring: adds special logic that merges multiple indices together

        Returns:

        """
        if old_name == new_name:
            return

        GLOBALS.p(f'renaming {old_name} to {new_name}')

        def rename_dictionary_entry(d: Dict[chr, Any], old_name: chr, new_name: chr):
            assert isinstance(d, dict)
            if old_name not in d:
                return
            if not allow_diagonal_rewiring:
                assert new_name not in d
            assert isinstance(old_name, str) and len(old_name) == 1
            assert isinstance(new_name, str) and len(new_name) == 1
            d[new_name] = d[old_name]
            del d[old_name]

        def rename_list_entry(ll, old_name, new_name):  # {len(l.count(old_name)}!=1
            assert isinstance(ll, list)
            if old_name not in ll:
                return
            if not allow_diagonal_rewiring:
                assert new_name not in ll
            assert isinstance(old_name, str)
            assert len(old_name) == 1
            assert isinstance(new_name, str)
            assert len(new_name) == 1

            assert ll.count(old_name) == 1, f"Found  {ll.count(old_name)} instances of {old_name} in {ll}"
            pos = ll.index(old_name)
            ll[pos] = new_name

        index_overlap = new_name in self.all_indices
        if not allow_diagonal_rewiring:
            assert not index_overlap, f"Renaming '{old_name}' to '{new_name}' but '{new_name}' already used in " \
                                      f"tensor {str(self)}"

        rename_list_entry(self.out_idx, old_name, new_name)
        rename_list_entry(self.in_idx, old_name, new_name)
        rename_list_entry(self.contracted_idx, old_name, new_name)
        # children_specs: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
        for i, index_spec in enumerate(self.children_specs):
            self.children_specs[i] = index_spec.replace(old_name, new_name)  # ab|c -> de|c
        #  _einsum_spec: str  # 'ij,jk->ik'

        if self._einsum_spec:
            self._einsum_spec = self._einsum_spec.replace(old_name, new_name)

        rename_dictionary_entry(self.idx_to_out_tensors, old_name, new_name)
        rename_dictionary_entry(self.idx_to_in_tensors, old_name, new_name)
        rename_dictionary_entry(self.idx_to_dim, old_name, new_name)
        rename_list_entry(self.out_idx, old_name, new_name)
        rename_list_entry(self.in_idx, old_name, new_name)
        rename_list_entry(self.contracted_idx, old_name, new_name)

    @property
    def all_indices(self):
        return tuple(self.out_idx + self.contracted_idx + self.in_idx)

    def _generate_unused_indices(self, other: 'TensorContraction', count=1):
        """Generate first count indices which aren't being used in current or other tensor"""
        assert count >= 1
        self_indices = set(self.in_idx + self.out_idx + self.contracted_idx)
        other_indices = set(other.in_idx + other.out_idx + other.contracted_idx)
        all_indices = self_indices.union(other_indices)
        assert len(all_indices.intersection(GLOBALS.all_indices)) == len(all_indices), "Using indices not in list of allowed indices " \
                                                                                       "(allowed in GLOBALS.all_indices)"
        largest_idx = max(self_indices.union(other_indices))
        largest_pos = GLOBALS.all_indices_list.index(largest_idx)
        # used_indices = self_indices.union(other_indices)
        # unused_indices = GLOBALS.all_indices.difference(used_indices)
        return GLOBALS.all_indices_list[largest_pos + 1:largest_pos + 1 + count]

    # def _generate_unused_indices(self, other: 'ContractibleTensor2', count=1):
    #     """Generate first count indices which aren't being used in current or other tensor"""
    #     assert count >= 1
    #
    #     self_indices = set(self.in_idx + self.out_idx + self.contracted_idx)
    #     other_indices = set(other.in_idx + other.out_idx + other.contracted_idx)
    #     used_indices = self_indices.union(other_indices)
    #     unused_indices = GLOBALS.all_indices.difference(used_indices)
    #     return tuple(sorted(unused_indices))[:count]

    @staticmethod
    def _symmetric_partition(dims) -> Tuple[Tuple[int], Tuple[int], int]:
        """Partitions a set of dimensions into two sets of equal size. (1,2,1,2) => (1,2), (1,2), 2"""
        d = len(dims)
        assert d % 2 == 0, f"can't partition {len(dims)} number of indices, it's an odd number"
        left = dims[:d // 2]
        right = dims[d // 2:]
        for (l, r) in zip(left, right):
            assert l == r, f"Can't symmetrically partition dims, dimensions don't match {left}!={right}"
        return left, right, d // 2

    @property
    def diag(self) -> 'TensorContraction':
        """Takes diagonal of the tensor."""
        bottom_dims, top_dims, split = self._symmetric_partition(self.in_dims)
        bottom_idx, top_idx = self.in_idx[:split], self.in_idx[split:]
        copy_tensors = []
        for bottom, top in zip(bottom_idx, top_idx):
            copy_tensors.append((bottom + top + '|' + bottom))
        return TensorContraction(self._original_specs, copy_tensors=copy_tensors)

    @property
    def T(self):
        # implement for matrices only for now
        # assert len(self.children_specs) == 1
        # assert len(self.children_data[0]) == 2
        new_specs = []
        for original_spec in self._original_specs:
            out_idx, in_idx = original_spec[0].split('|')
            new_idx_spec = in_idx + '|' + out_idx

            assert len(original_spec[1].shape) <= 2, "Don't support transposing tensors with rank above 2"
            if len(original_spec[1].shape) == 2:
                new_data = original_spec[1].T
            else:
                new_data = original_spec[1]

            if len(original_spec) == 3:
                (spec, data, name) = original_spec
                new_tensor_spec = (new_idx_spec, new_data, name)
            else:
                assert len(original_spec) == 2
                (spec, data) = original_spec
                new_tensor_spec = (new_idx_spec, new_data)
            new_specs.insert(0, new_tensor_spec)
        return TensorContraction(new_specs)
        # return TensorContraction([(self.children_specs[0], self.children_data[0].T, self.children_labels[0])])

    @property
    def trace(self):
        bottom_dims, top_dims, split = self._symmetric_partition(self.in_dims)
        bottom_idx, top_idx = self.in_idx[:split], self.in_idx[split:]
        copy_tensors = []
        for bottom, top in zip(bottom_idx, top_idx):
            copy_tensors.append((bottom + top + '|'))
        return TensorContraction(self._original_specs, copy_tensors=copy_tensors)

    def __mul__(self, other: 'TensorContraction') -> 'TensorContraction':
        """Contraction operation"""

        assert isinstance(other, TensorContraction)
        GLOBALS.p(f'contracting {self} with {other}')
        #        self._check_indices_sorted()
        #        other._check_indices_sorted()

        # left = ContractibleTensor2(self._original_specs, tag=f'temp_{self.label}')
        left = self

        # clone the right tensor to make use of index relabeling without corrupting given right tensor
        right = TensorContraction(list(other._original_specs), label=f'cloned_{other.label}')

        # this can potentially be relaxed to make contractions commutative
        # however, all current applications only need left-to-right order, and this was the only case that's well tested, hence require it
        if not GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES:
            assert len(left.in_idx) >= len(
                right.out_idx), f"only allow partial contraction on left tensor, right tensor must contract all output indices, contracting {left.ricci_out} with {right.ricci_out}"

        # rename indices of right to avoid clashes
        if len(right.contracted_idx + right.in_idx):
            GLOBALS.p('before step 1 rename')
            GLOBALS.p(right.children_specs)

            max_renames = len(right.contracted_idx + right.in_idx)
            new_indices = left._generate_unused_indices(other=right, count=max_renames)
            rename_count = 0
            left_uncontracted_in_idx = left.in_idx[len(right.out_idx):]
            taken_indices = set(left.in_idx + left.contracted_idx + left.out_idx)  # these indices are used by LEFT tensor
            for idx in right.contracted_idx + right.in_idx:
                # rename all indices of RIGHT that conflict with LEFT unless they are in right's out index set (happens for diagonal)
                if idx in taken_indices:
                    right._rename_index(idx, new_indices[rename_count])
                    rename_count += 1

        # match right's out indices to left's in indices
        # left tensor may have more indices than right, this happens in Hessian-vector product
        GLOBALS.p('before step 2 rename', left.children_specs, right.children_specs)
        if not GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES:
            left_contracted = left.in_idx[:len(right.out_idx)]  # contract these in-indices of LEFT with all out-indices of RIGHT
        else:
            # find in_idx corresponding to the tensor with the most in-indices

            tensors_with_in_indicies = {}
            largest_in_tensor_id = -1
            largest_in_tensor_in_count = -1
            for tensor_id, tensor_spec in enumerate(self.children_specs):
                out_idx, in_idx = tensor_spec.split('|')
                if set(in_idx).intersection(left.in_idx):
                    tensors_with_in_indicies[tensor_id] = len(in_idx)
                    if len(in_idx) > largest_in_tensor_in_count:
                        largest_in_tensor_id = tensor_id
                        largest_in_tensor_in_count = len(in_idx)
            assert largest_in_tensor_id >= 0

            tensor_spec = self.children_specs[largest_in_tensor_id]
            out_idx, in_idx = tensor_spec.split('|')
            assert len(set(in_idx).intersection(left.in_idx)) >= len(
                right.out_idx), f"not enough in indices to contract on tensor {largest_in_tensor_id}"
            left_contracted = list(set(in_idx).intersection(left.in_idx))[:len(right.out_idx)]
            # if not in_idx_ranks.keys():
            #     # fall back on previous method, which works for diagonal tensors
            #     left_contracted = left.in_idx[:len(right.out_idx)]
            # else:
            #     top_rank = max(in_idx_ranks.keys())
            #     assert len(in_idx_ranks[top_rank]) >= len(right.out_idx), "Couldn't find tensor to contract with right"
            #     left_contracted = in_idx_ranks[top_rank][:len(right.out_idx)]

        GLOBALS.p(f'matching left {left_contracted} to right {right.out_idx}')
        for left_idx, right_idx in zip(left_contracted, right.out_idx):
            right._rename_index(right_idx, left_idx)
        GLOBALS.p('after step 2 rename')
        GLOBALS.p(right.children_specs)

        # TODO: here (add ..)
        new_specs = self._transpose_specs(left.children_specs + right.children_specs, left.children_data + right.children_data,
                                          left.children_labels + right.children_labels)
        result = TensorContraction(new_specs, label=f"{self.label}*{other.label}")
        GLOBALS.p(f'contracting {self.label} and {other.label}')
        GLOBALS.p(','.join(self.children_specs) + ' * ' + ','.join(other.children_specs) + ' = ' + ','.join(result.children_specs))
        return result

    @staticmethod
    def _transpose_specs(children_specs, tensors, tensor_labels):
        return list((spec, tensor, label) for (spec, tensor, label) in zip(children_specs, tensors, tensor_labels))

    @staticmethod
    def from_dense_vector(x: torch.Tensor, label: str = None, idx: str = None) -> 'TensorContraction':
        """Creates StructuredTensor object corresponding to given dense vector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        idx0 = GLOBALS.idx0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord(idx0), ord(idx0) + len(x.shape)))
        return TensorContraction.__legacy_init__([idx + '|'], [x], label)

    @staticmethod
    def from_dense_covector(x: torch.Tensor, label: str = None, idx: str = None) -> 'TensorContraction':
        """Creates StructuredTensor object corresponding to given dense covector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        idx0 = GLOBALS.idx0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord(idx0), ord(idx0) + len(x.shape)))
        return TensorContraction.__legacy_init__(['|' + idx], [x], label)

    # TODO(y): label/tag parameters fix, half of the methods start with tensor, another half with index
    @staticmethod
    def from_dense_matrix(x: torch.Tensor, label: str = None) -> 'TensorContraction':
        """Creates StructuredTensor object treating it as linear map (1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        idx0 = GLOBALS.idx0
        idx1 = chr(ord(idx0) + 1)
        return TensorContraction.__legacy_init__([idx0 + '|' + idx1], [x], label)

    @staticmethod
    def from_dense_quadratic_form(x: torch.Tensor, label: str = None) -> 'TensorContraction':
        """Creates StructuredTensor object treating it as linear map (1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        idx0 = GLOBALS.idx0
        idx1 = chr(ord(idx0) + 1)
        return TensorContraction.__legacy_init__(['|' + idx0 + idx1], [x], label)

    @staticmethod
    def from_dense_tensor(tensor_spec: str, x: torch.Tensor, label: str = None) -> 'TensorContraction':
        return TensorContraction([(tensor_spec, x, label)])

    @staticmethod
    def from_diag_matrix(x: torch.Tensor, label: str = None) -> 'TensorContraction':
        """Creates ContractibleTensor corresponding to diagonal matrix"""
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        assert x.shape[0] > 0
        idx0 = GLOBALS.idx0
        return TensorContraction.__legacy_init__([idx0 + '|' + idx0], [x], label)

    @property
    def value(self) -> torch.Tensor:
        if not self._einsum_spec:
            # manually materialize for diagonal matrix case, since einsum doesn't support i->ii syntax
            assert len(self.in_idx) == 1, "Only support diagonal rank-2 diagonal tensors"
            assert len(self.children_data) == 1, f"Only support diagonal rank-2 single diagonal matrix, got {len(self.children_data)}"
            return torch.diag(self.children_data[0])

        # hack to deal with diagonal tensors, not needed for einsum optimizer
        ein_in, ein_out = self._einsum_spec.split('->')
        new_terms = []
        for term in ein_in.split(','):
            if len(term) == 2 and term[0] == term[1]:
                new_term = term[0]
            else:
                new_term = term
            new_terms.append(new_term)

        new_einsum_spec = ','.join(new_terms) + '->' + ein_out
        if self.copy_tensors is None:  # have new logic for copy tensors, this is old logic
            if new_einsum_spec != self._einsum_spec:
                GLOBALS.p("Warning, diagonal hack")
                return torch.einsum(new_einsum_spec, *self.children_data)
        return torch.einsum(self._einsum_spec, *self.children_data)


class ZeroTensor(Tensor):
    """Arbitrary shape tensor of zeros"""

    def in_dims(self):
        return ()

    def out_dims(self):
        return ()


# zero = ZeroTensor()
# Zero = ZeroTensor()  # TODO(y) remove one of the zeros


class Scalar(Tensor, ABC):
    """Scalar, empty output and input indices"""

    def out_dims(self):
        return ()

    def in_dims(self):
        return ()


class Vector(Tensor, ABC):
    """pure contravariant Tensor, upper (output) indices only"""

    
    def out_dims(self) -> Tuple[int]:
        pass


class Covector(Tensor, ABC):
    """pure covariant Tensor, lower (input) indices only"""

    
    def in_dims(self) -> Tuple[int]:
        pass


class LinearMap(Tensor, ABC):
    """mixed Tensor, one set of upper and one set of lower"""

    
    def out_dims(self) -> Tuple[int]:
        pass

    
    def in_dims(self) -> Tuple[int]:
        pass


class QuadraticForm(Tensor, ABC):
    # this must return ()
    
    def out_dims(self) -> Tuple[int]:
        pass

    
    def in_dims(self) -> Tuple[int]:
        pass


class SymmetricBilinearMap(Tensor, ABC):
    """Symmetric bilinear map. Two sets of input indices with equal dimensions.
    TODO(y): enforce this somehow"""

    
    def out_dims(self) -> Tuple[int]:
        pass

    
    def in_dims(self) -> Tuple[int]:
        pass


class IdentityLinearMap(Tensor):
    """tensor representing identity linear map"""

    def in_dims(self):
        return ()

    def out_dims(self):
        return ()


##################################################
# Functions
##################################################

class FunctionSharedImpl:
    name: str  # name of the layer, used for human-readable representation

    base_name: str  # name of original function, before differentiation
    order: int  # derivative order

    def __init__(self, name=None, base_name=None, order=0):
        """Creating name for function:
        if name is specified, use that
        if name is not specified, automatically construct it for derivatives
        """

        self.order = order
        self.base_name = base_name
        type_name = type(self).__name__

        # if name not provided, fill it in automatically, see diag02
        if name is None:
            if order > 0 or base_name is not None or isinstance(self, LinearizedFunction):
                # make sure that all functions with "order>0" property are linearized functions
                assert order > 0
                assert isinstance(self, LinearizedFunction)
                assert name is None or base_name is None, f"Provided both name ({name}) and base_name ({base_name})"
                if base_name is None:
                    assert type_name.startswith('D_')
                    name = 'D_' * (self.order - 1) + type_name
                else:
                    name = 'D_' * self.order + base_name
            else:
                name = type_name

            # dedup
            count = GLOBALS.function_count.get(name, 0)
            GLOBALS.function_count[name] = count + 1
            name = name if count == 0 else f"{name}{count:02d}"
        if name not in ['@', '+', '*']:  # don't save automatically created composite functions
            # assert name not in GLOBALS.function_dict, f"Function {name} has already been created"
            GLOBALS.function_dict[name] = self
        self.name = name

    # def construct_derivative_layer_name(self, base_name) -> str:
    #     """DLinearLayer, DDLinearLayer, DDDLinearLayer, etc"""
    #
    #     type_name = type(self).__name__
    #     assert type_name.startswith("D_") or type_name.startswith("DD_") or type_name.startswith("DDD_")  # first 3 derivatives for now
    #     assert isinstance(self, LinearizedFunction) and hasattr(self, "order")
    #     assert getattr(self, 'order') >= 1
    #     if base_name.startswith('D_'):
    #         return "D" * (getattr(self, "order") - 1) + base_name
    #     else:
    #         return "D" * (getattr(self, "order") - 1) + 'D_' + base_name

    def __add__(self, other: 'Function'):
        assert isinstance(other, Function)
        if isinstance(other, ZeroFunction):
            return self
        if isinstance(self, FunctionAddition):
            return FunctionAddition(self.children + [other])
        else:
            return FunctionAddition([self, other])

    # contraction
    def __mul__(self, other: 'Function'):
        assert isinstance(other, Function)
        if isinstance(other, ZeroFunction):
            return ZeroFunction()
        if isinstance(other, IdentityFunction):
            return self
        if isinstance(self, FunctionContraction):
            return FunctionContraction(self.children + [other])
        else:
            return FunctionContraction([self, other])

    # composition
    def __matmul__(self, other: 'Function'):
        if GLOBALS.switch_composition_order:
            return other.__rmatmul__(self)
        else:
            # don't merge Function Composition chains, this complicates caching.
            if GLOBALS.enable_memoization:
                return MemoizedFunctionComposition([self, other])
            else:
                return UnmemoizedFunctionComposition([self, other])

    #            if isinstance(self, FunctionComposition):
    #                return FunctionComposition(self.children + [other])
    #            else:
    #                return FunctionComposition([self, other])

    def __rmatmul__(self, other: 'Function'):
        GLOBALS.p("Calling R-matmul")
        if isinstance(self, MemoizedOrUnemoizedFunctionComposition):
            if GLOBALS.enable_memoization:
                return MemoizedFunctionComposition([other] + self.children)
            else:
                return UnmemoizedFunctionComposition([other] + self.children)
        else:
            if GLOBALS.enable_memoization:
                return MemoizedFunctionComposition([other, self])
            else:
                return UnmemoizedFunctionComposition([other, self])

    @property
    def human_readable(self):
        """Return human-readable representation of f.

        ie, DLeastSquares@LinearLayer, (DDLeastSquares@LinearLayer)*DLinearLayer

        """

        if isinstance(self, CompositeFunction):
            child_str = self.name.join(str(c) for c in self.children)
            assert len(self.children) > 0, "found composition with one child or less"
            # return f'({self.name}:{child_str})'  # (1+2)
            return f'({child_str})'  # (1+2)
        else:
            return self.name if hasattr(self, "name") else f"(Unnnamed type(self).__name__)"

    def __str__(self):
        return self.human_readable

    def __repr__(self):
        return str(self)


class Function(ABC):
    """Differentiable function"""

    
    def __call__(self, t: 'Tensor'):
        pass

    # 
    # def in_dims(self):
    #    """Input (lower) dimensions"""
    #    pass

    # 
    # def out_dims(self):
    #    """Output (upper) dimensions"""
    #    pass


class CompositeFunction(Function, FunctionSharedImpl, ABC):
    """Function defined as a combination of AtomicFunction objects using +, *, @"""
    # TODO(y): use tuple instead of list

    children: List[Function]

    ## TODO(y) drop dimensions? These are are only needed at tensor level
    # def out_dims(self):
    #    pass

    # def in_dims(self):
    #    pass


class FunctionAddition(CompositeFunction):
    def __init__(self, children: List['Function']):
        # Must have two children. Otherwise, it's harder to tell if two functions are the same
        # ie, FunctionAddition([f]) == FunctionContraction([f])
        super().__init__(name='+')
        # self.name = '+'
        assert len(children) >= 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[0](t)
        for c in self.children[1:]:
            result = result + c(t)
        return result


def make_function_addition(children) -> Function:
    children = [c for c in children if not isinstance(c, ZeroFunction)]
    if len(children) == 0:
        return ZeroFunction()
    elif len(children) == 1:
        return children[0]
    else:
        return FunctionAddition(children)


class FunctionContraction(CompositeFunction):
    def __init__(self, children: List['Function']):
        super().__init__(name='*')
        assert len(children) >= 2
        # self.name = '*'
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[0](t)
        for c in self.children[1:]:
            child_result = c(t)
            result = result * child_result
        return result


def make_function_contraction(children) -> Function:
    for c in children:
        if isinstance(c, ZeroFunction):
            return ZeroFunction()
    if len(children) == 0:
        return IdentityFunction()
    elif len(children) == 1:
        return children[0]
    else:
        return FunctionContraction(children)


class MemoizedOrUnemoizedFunctionComposition(CompositeFunction):
    def is_arg_bound(self):
        pass


class MemoizedFunctionComposition(MemoizedOrUnemoizedFunctionComposition):
    """Represents a composition of functions with memoization
    Unbound call, ie f@g@h, can be used as intermediate result for constructing a composition
    Bound call, ie (f@g@h)(x), at this point it is frozen and can't be modified.
    """

    children: List[Any]  # List[Function] fix: forward references
    parent: 'MemoizedFunctionComposition'
    arg: Any

    def __init__(self, children, parent=None, arg=None):
        super().__init__(name='@')

        self.arg = arg
        self.parent = parent
        self.children = children
        self._saved_outputs = [None] * (len(children) + 1)

        # if creating a sub-composition, extra sanity check that the nodes we are using
        # are already pointing to the parent composition
        # for child in children:
        #     if parent:
        #         #  assert child.parent == parent
        #         pass
        #     else:
        #         if hasattr(child, 'parent') and child.parent is not None:
        #             if child.parent == self:
        #                 pass  # all is good, still pointing to current parent
        #             else:
        #                 # special case for derivatives, we create new chains like D(f4) f3 f2 f1
        #                 if self.children and type(self.children[0]).__name__.startswith('D'):
        #                     pass
        #                 else:
        #                     assert False, f"Warning, Node {child} already has parent {child.parent} which is not current parent {self}"
        #         child.parent = self

    def __matmul__(self, other):
        assert self.arg is None, "Can't combine compositions with bound parameters"
        if isinstance(other, Function):
            # don't merge Function Composition chains, this complicates caching.
            # return MemoizedFunctionComposition(self.children + [other])
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented

    # only allow simple slicing
    def __getitem__(self, s):
        assert self.arg is not None, f"Can't slice composition chain {self}{s} without binding argument first"
        if isinstance(s, slice):
            if isinstance(s.step, int):
                assert s.step == 1
            # error_msg = "this case hasn't been tested, for now only single level of parent  redirection is allowed"
            # assert self.parent is None, error_msg
            backlink = self if self.parent is None else self.parent
            assert s.stop is None
            assert len(self.children[s.start:]) > 0, f"only have {len(self.children)} members of composition, " \
                                                     f"attempted to start at {s.start} "
            sub_composition = MemoizedFunctionComposition(self.children[s.start:], backlink)
            sub_composition._bind(self.arg)
            return sub_composition
        else:
            assert False, "use [i:i+1] slicing instead of [i] is ambiguous"
            # assert isinstance(s, int)
            # return self.children[s]

    def _bind(self, arg):
        GLOBALS.p('binding ', arg)
        self.arg = arg
        self._saved_outputs[len(self.children)] = arg

    def is_arg_bound(self):
        return hasattr(self, 'arg') and self.arg is not None

    def memoized_compute(self, node) -> Optional[torch.Tensor]:
        """Composition([h3,h2,h1]): memoized_compute(h2) computes everything up to h2"""
        GLOBALS.p(f'{node}: {self}')
        GLOBALS.p(f"{node} in {self}: {node in self.children}")
        GLOBALS.p(
            f"### {node} in {self}, have child: {node in self.children} have parent: {self.parent is not None}, {self.parent is not None and 'D_' not in self.parent.human_readable:}")
        # assert self.arg is not None, f"argument not bound in {self}, call _bind first"
        # assert id(self._saved_outputs[len(self.children)]) == id(self.arg)

        if self.parent is not None:
            # terrible hack, special handling for new compositions with derivative terms
            if "D_" not in self.parent.human_readable:
                GLOBALS.p(f"### {self} deferring to parent {self.parent}")
                assert isinstance(self.parent, MemoizedFunctionComposition)
                return self.parent.memoized_compute(node)

        # assert node in self.children, f"Trying to compute {node} but it's not in Composition"
        if node in self.children:
            idx = self.children.index(node)

            # last_saved gives position of function whose output has been cached
            # we treat "x" as a dummy function which returns itself, so
            # last_cached == len(children) means just the output of "x" was cached
            for last_cached in range(len(self.children)):
                if self._saved_outputs[last_cached] is not None:
                    break
            else:
                last_cached = len(self.children)

            GLOBALS.p(f'found saved output of {last_cached} node')
            for i in range(last_cached - 1, idx - 1, -1):
                if i == len(self.children):
                    assert id(self._saved_outputs[last_cached]) == id(self.arg)
                    continue

                if not isinstance(self.children[i], MemoizedFunctionComposition):
                    GLOBALS.p(f"### {self} calling compute on {self.children[i]}")

                result = self.children[i](self._saved_outputs[i + 1])
                self._saved_outputs[i] = result
                GLOBALS.p('saving output of ', i)

            return self._saved_outputs[idx]
        else:
            # try to defer to child composition, ie
            # value is requested for U in [D_lsqr @ (U@relu@W)], this defers computation of U to child (U@relu@W)
            for child in self.children:
                if isinstance(child, MemoizedFunctionComposition):
                    try:
                        return child.memoized_compute(node)
                    except RuntimeError:
                        continue
        raise RuntimeError(f"couldn't compute {node} in Composition {self}")

    def __call__(self, arg):
        if self.parent is not None:
            assert isinstance(self.parent, MemoizedFunctionComposition)
            if self.parent.arg is not None:
                if arg is not None:
                    assert id(arg) == id(self.parent.arg)
            # else:
            #    self.parent._bind(arg)
            return self.parent.memoized_compute(self.children[0])

        if self.arg is None:
            self._bind(arg)
        else:
            assert id(self.arg) == id(arg), "Reusing same composition for multiple args"

        return self.memoized_compute(self.children[0])

    @property
    def value(self):
        if self.parent:
            result = self.__call__(self.parent.arg)
        else:
            assert self.arg, "Trying to get value of function composition, but arg has not been supplied yet"
            result = self.__call__(self.arg)
        if hasattr(result, 'value'):
            return result.value


# TODO(y): rename to FunctionNesting? compositvefunction/functioncomposition clash
class UnmemoizedFunctionComposition(MemoizedOrUnemoizedFunctionComposition):
    def __init__(self, children: List['Function']):
        super().__init__(name='@')
        # self.name = '@'
        # assert len(children) >= 2
        self.children = children
        self.arg = None

    def __getitem__(self, s):
        if isinstance(s, slice):
            if isinstance(s.step, int):
                assert s.step == 1
            assert s.stop is None
            assert len(self.children[s.start:]) > 0, f"only have {len(self.children)} members of composition, " \
                                                     f"attempted to start at {s.start} "
            return UnmemoizedFunctionComposition(self.children[s.start:])
        else:
            assert False, "use [i:i+1] slicing instead of [i] is ambiguous"
            # assert isinstance(s, int)
            # return self.children[s]

    def __call__(self, t: 'Tensor'):
        result = self.children[-1](t)
        for c in reversed(self.children[:-1]):
            result = c(result)
        return result

    def _bind(self, x):  # for compatibility with MemoizedFunctionComposition
        pass

    def is_arg_bound(self):
        return hasattr(self, 'arg') and self.arg is not None


# FunctionComposition = UnmemoizedFunctionComposition
# FunctionComposition = MemoizedFunctionComposition

def make_function_composition(children, arg=None):
    if len(children) == 0:
        return ZeroFunction()
    elif isinstance(children[0], ZeroFunction):
        return ZeroFunction()
    elif len(children) == 1:
        return children[0]
    else:
        if GLOBALS.enable_memoization:
            return MemoizedFunctionComposition(children, arg=arg)
        else:
            return UnmemoizedFunctionComposition(children)


class AtomicFunction(Function, FunctionSharedImpl):
    @property
    def d1(self):
        return self.d(1)

    
    def d(self, order: int):
        pass


class LinearizedFunction(ABC):
    """This represents a function which outputs Tensor objects."""
    pass


##################################################
# Operators
##################################################

class Operator(ABC):
    other: 'Function'

    def __call__(self, other):
        pass

    # def __matmul__(self, other):
    #    return OperatorComposition([self, other])


# We only have 1 Operator for now. Later could add Trace operator, but don't need to separate
# composition logic for now
class OperatorComposition:
    children: List[Operator]

    def __init__(self, children):
        self.children = children

    # composition operation
    def __matmul__(self, other):
        if isinstance(other, Operator):
            return OperatorComposition([self, other])
        else:
            return NotImplemented


class D_(Operator):
    """Differentiation of arbitrary order. IE D_(1) for derivative, D_(2) for Hessian etc"""
    order: int

    def __init__(self, _dont_use_order=1):
        assert _dont_use_order >= 1
        self.order = _dont_use_order

    def __call__(self, other: Function) -> Function:
        # atomic function, defer to existing derivative implementation
        assert isinstance(other, Function), f"D works on Function type, instead was given {type(Function)}"
        if isinstance(other, AtomicFunction):
            return other.d(self.order)

        # addition rule
        elif isinstance(other, FunctionAddition):
            return make_function_addition(self(c) for c in other.children)

        # product rule
        elif isinstance(other, FunctionContraction):
            add_children: List[Function] = []
            for (i, c1) in enumerate(other.children):
                mul_children: List[Function] = []
                dc1 = D(c1)
                GLOBALS.p("differentiating ", c1, " got ", D(c1))
                for (j, c2) in enumerate(other.children):
                    if i == j:
                        mul_children.append(dc1)
                    else:
                        mul_children.append(c2)
                add_children.append(make_function_contraction(mul_children))
            return make_function_addition(add_children)

        # chain rule
        elif isinstance(other, MemoizedOrUnemoizedFunctionComposition):
            if isinstance(other, MemoizedFunctionComposition):
                assert other.is_arg_bound()
            mul_children1 = []  # old way
            for (i, c1) in enumerate(other.children):
                if i + 1 == len(other.children):  # last term
                    mul_children1.append(D(c1))
                else:
                    mul_children1.append(make_function_composition([D(c1)] + [other[i + 1:]], arg=other.arg))

            return make_function_contraction(mul_children1)

        else:
            assert False, f"Unknown node type: {other}"


D = D_(_dont_use_order=1)
dont_use_D2 = D_(_dont_use_order=2)  # this operator only works correctly on leaf nodes


def diag(t: TensorContraction):
    return t.diag


def trace(t: TensorContraction):
    return t.trace


class OldContractibleTensor(Tensor, ABC):
    """
    Original implementation, only supports 1 and 2 index contractions
    """

    @property
    def out_dims(self) -> Tuple[int]:
        return 42,

    @property
    def in_dims(self) -> Tuple[int]:
        return 42,

    def out_idx(self):
        upper = ''
        offset = 0
        # TODO(y): Covector does not have out_dims defined, test what happens
        # TODO(y): replace out_dims with property call
        if hasattr(self, 'out_dims'):
            upper = ''.join(chr(ord(GLOBALS.idx0) + offset + i) for i in range(len(self.out_dims)))
        return upper

    def in_idx(self):
        # TODO(y) replace with properties
        lower = ''
        offset = 0
        if hasattr(self, 'in_dims'):
            lower = ''.join(chr(ord(GLOBALS.idx0) + offset + i) for i in range(len(self.in_dims)))
        return lower

    def all_idx(self, offset=0):
        """Generate string corresponding to upper,lower indices, with offset.
        IE for tensor with 2 upper indices 'ab' and 1 lower index 'c'
        IE, offset 0: "abc"
            offset 1: "bcd", etc"""

        upper, lower = '', ''
        if hasattr(self, 'out_dims'):
            upper = ''.join(chr(ord(GLOBALS.idx0) + offset + i) for i in range(len(self.out_dims)))

        offset += len(upper)
        if hasattr(self, 'in_dims'):
            lower = ''.join(chr(ord(GLOBALS.idx0) + offset + i) for i in range(len(self.in_dims)))
        return upper + lower, upper, lower

    def __mul__(self, other):
        # GLOBALS.p('other is ', other)
        assert isinstance(other, OldContractibleTensor), f"contracting tensor with {type(other)}"

        t1 = self
        t2 = other

        # assert isinstance(t1, DenseLinear) and isinstance(t2, DenseVector), "contraction tested only for
        # matrix@vector " return self.W * x

        assert t1.in_dims == t2.out_dims  # ij,j -> i
        (t1_idx, t1_idx_out, t1_idx_in) = t1.all_idx()
        (t2_idx, t2_idx_out, t2_idx_in) = t2.all_idx(offset=len(t1.out_idx()))

        t1_set = set(t1_idx)
        t2_set = set(t2_idx)
        contracted_set = t1_set.intersection(t2_set)
        result_set = t1_set.union(t2_set).difference(contracted_set)
        result_idx = ''.join(sorted(list(result_set)))

        einsum_str = f"{t1_idx},{t2_idx}->{result_idx}"
        data = torch.einsum(einsum_str, t1.value, t2.value)

        # figure out new input, output indices, create corresponding object
        out_idx = result_set.intersection(set(t1_idx_out + t2_idx_out))
        in_idx = result_set.intersection(set(t1_idx_in + t2_idx_in))
        if out_idx and not in_idx:
            assert len(out_idx) == 1, "don't support multi-index yet"
            return DenseVector(data)
        elif in_idx and not out_idx:
            assert len(in_idx) == 1, "don't support multi-index yet"
            return DenseCovector(data)
        elif in_idx and out_idx:
            assert len(in_idx) == len(out_idx) == 1, "don't support multi-index yet"
            return DenseLinear(data)
        elif not in_idx and not out_idx:
            assert data.shape == ()
            return DenseScalar(data)

    def __str__(self):
        return str(self.value)


class DenseScalar(Scalar):
    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 0
        self._value = value

    @property
    def in_dims(self):
        return ()

    @property
    def out_dims(self):
        return ()

    @property
    def value(self):
        return self._value


class DenseVector(Vector, OldContractibleTensor):
    _value: torch.Tensor
    _out_dims: Tuple[int]

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 1
        assert value.shape[0] > 0
        self._out_dims = value.shape
        self._value = value

    @property
    def in_dims(self):
        return ()

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def value(self):
        return self._value

    @property
    def T(self):
        return DenseCovector(self._value)


class DenseCovector(Covector, OldContractibleTensor):
    _value: torch.Tensor
    _in_dims: Tuple[int]

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 1
        assert value.shape[0] > 0
        self._in_dims = value.shape
        self._value = value

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return ()

    @property
    def value(self):
        return self._value

    @property
    def T(self):
        return DenseCovector(self._value)


class DenseSymmetricBilinear(SymmetricBilinearMap, OldContractibleTensor):
    """Symmetric bilinear map represented with a rank-3 tensor"""

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 3
        assert value.shape[0] > 0
        assert value.shape[1] > 0
        assert value.shape[2] > 0
        assert value.shape[1] == value.shape[2]
        self._out_dims = (value.shape[0],)
        self._in_dims = (value.shape[1],)
        self._value = value

    @property
    def out_dims(self) -> Tuple[int]:
        return self._out_dims

    @property
    def in_dims(self) -> Tuple[int]:
        return self._in_dims

    @property
    def value(self):
        return self._value


class DenseQuadraticForm(QuadraticForm, OldContractibleTensor):
    """Symmetric bilinear map represented with a rank-2 tensor"""

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 2
        assert value.shape[0] > 0
        assert value.shape[1] > 0
        assert value.shape[0] == value.shape[1]
        self._out_dims = ()
        self._in_dims = (value.shape[1],)
        self._value = value

    @property
    def out_dims(self) -> tuple:
        return self._out_dims

    @property
    def in_dims(self) -> Tuple[int]:
        return self._in_dims

    @property
    def value(self):
        return self._value


class DenseLinear(LinearMap, OldContractibleTensor):
    """Symmetric linear map represented with a rank-2 tensor"""

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 2
        assert value.shape[0] > 0
        assert value.shape[1] > 0
        self._out_dims = (value.shape[0],)
        self._in_dims = (value.shape[1],)
        self._value = value

    @property
    def out_dims(self) -> Tuple[int]:
        return self._out_dims

    @property
    def in_dims(self) -> Tuple[int]:
        return self._in_dims

    @property
    def value(self):
        return self._value


# noinspection PyMissingConstructor
class OldLeastSquares(AtomicFunction):
    """Least squares loss"""

    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = ()

    def __call__(self, x: Tensor):
        x = x.value
        return DenseScalar((x * x).sum() / 2)

    def d(self, order=1):
        return OldDLeastSquares(dim=self._in_dims[0], order=order)

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class LeastSquares(AtomicFunction):
    """Least squares loss"""

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x: TensorContraction):
        x = x.value
        GLOBALS.increment_global_forward_flops(1)
        return DenseScalar((x * x).sum() / 2)

    def d(self, order=1):
        return D_LeastSquares(order=order, base_name=self.human_readable)

    @property
    def in_dims(self):
        return -43

    @property
    def out_dims(self):
        return -45

    # def __matmul__(self, other):
    #     if isinstance(other, AtomicFunction):
    #         return FunctionComposition([self, other])
    #     else:
    #         return NotImplemented


class D_LeastSquares(AtomicFunction, LinearizedFunction):
    """Derivatives of LeastSquares"""

    def __init__(self, name=None, base_name=None, order: int = 1):
        super().__init__(name=name, base_name=base_name, order=order)

    def __call__(self, x: TensorContraction) -> TensorContraction:
        assert self.order <= 2, "third and higher order derivatives not implemented"
        GLOBALS.increment_global_forward_flops(1)

        if self.order == 1:
            return x.T
        elif self.order == 2:
            # GLOBALS.p("")
            # assert False, f"sys.exit() {GLOBALS.DEBUG_HESSIAN}"
            assert len(x.out_dims) == 1
            if GLOBALS.FULL_HESSIAN:
                return TensorContraction.from_dense_quadratic_form(torch.eye(x.out_dims[0]))
            else:
                # return Hessian subspace using Rademacher variables
                d = x.out_dims[0]
                x00 = (torch.randint(0, 2, (d,)) * 2 - 1).float()
                return TensorContraction([('|a', x00), ('|b', x00)])

    @property
    def d1(self):
        return self.d(1)

    def d(self, order=1):
        return D_LeastSquares(order=self.order + order)

    @property
    def in_dims(self):
        return -47

    @property
    def out_dims(self):
        return -48


# noinspection PyMissingConstructor
class OldDLeastSquares(AtomicFunction, LinearizedFunction):
    """Derivatives of LeastSquares"""

    def __init__(self, dim: int, order: int = 1):
        self._in_dims = (dim,)
        self._out_dims = ()
        self.order = order

    def __call__(self, x: DenseVector):
        assert self.order <= 2, "third and higher order derivatives not implemented"
        n = self._in_dims[0]

        if self.order == 1:
            return x.T
        elif self.order == 2:
            return DenseQuadraticForm(torch.eye(n))
        # three-dimensional identity tensor, does not exist in numpy
        elif self.order == 3:
            assert False, "TODO: wrap this into proper rank-3 tensor"
            # x = torch.einsum('ij,jk->ijk', torch.eye(n), torch.eye(n))

    @property
    def d1(self):
        return self.d(1)

    def d(self, order=1):
        return OldDLeastSquares(dim=self._in_dims[0], order=self.order + order)

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


import torch.nn.functional as F


# noinspection PyMissingConstructor
class OldRelu(AtomicFunction):
    """One dimensional relu"""

    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = (dim,)

    def __call__(self, x: Tensor):
        x = x.value
        return DenseVector(F.relu(x))

    def d(self, order=1):
        return OldDRelu(self._in_dims[0], order=order)

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


# noinspection PyMissingConstructor
class OldDRelu(AtomicFunction, LinearizedFunction):
    """Derivatives of relu"""

    def __init__(self, dim: int, order: int = 1):
        self._in_dims = (dim,)
        self._out_dims = (dim,)
        self.order = order

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def in_dims(self):
        return self._in_dims

    def d(self, order=1):
        return ZeroFunction()

    def __call__(self, x):
        if self.order == 1:
            x = x.value
            return DenseLinear(torch.diag((x > torch.tensor(0)).float()))
            #            return TensorContraction.from_diag_matrix((x > torch.tensor(0)).float())

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class Relu(AtomicFunction):
    """One dimensional relu"""

    def out_dims(self):
        pass

    def in_dims(self):
        pass

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x: TensorContraction):
        assert isinstance(x, TensorContraction)
        GLOBALS.increment_global_forward_flops(1)
        x = x.value
        return TensorContraction.from_dense_vector(F.relu(x))

    def d(self, order=1):
        if order == 1:
            return D_Relu(order=order, base_name=self.human_readable)
        else:
            return ZeroFunction()

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return FunctionComposition([self, other])
        else:
            return NotImplemented


class D_Relu(AtomicFunction, LinearizedFunction):
    """Derivatives of relu"""

    def __init__(self, name=None, base_name=None, order: int = 1):
        # super.__init__(name=name, base_name=base_name, order=order)
        super().__init__(name=name, base_name=base_name, order=order)

    def d(self, order=1):
        return ZeroFunction()

    def __call__(self, x: TensorContraction) -> TensorContraction:
        GLOBALS.increment_global_forward_flops(1)
        if self.order == 1:
            x = x.value
            return TensorContraction.from_diag_matrix((x > torch.tensor(0)).float())

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            if GLOBALS.enable_memoization:
                return MemoizedFunctionComposition([self, other])
            else:
                return UnmemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


# noinspection PyMissingConstructor
class OldSigmoid(AtomicFunction):
    """One dimensional relu"""

    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = (dim,)

    def __call__(self, x: Tensor):
        x = x.value
        return DenseVector(torch.sigmoid(x))

    def d(self, order=1):
        return OldDSigmoid(self._in_dims[0], order=order)

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return FunctionComposition([self, other])
        else:
            return NotImplemented


# noinspection PyMissingConstructor
class OldDSigmoid(AtomicFunction, LinearizedFunction):
    """Derivatives of sigmoid"""

    def __init__(self, dim: int, order: int = 1):
        assert order >= 1
        self._in_dims = (dim,)
        self._out_dims = (dim,)
        self.order = order

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def in_dims(self):
        return self._in_dims

    def d(self, order=1):
        return OldDSigmoid(self._in_dims[0], order=self.order + order)

    def __call__(self, x: Tensor) -> OldContractibleTensor:
        x = x.value
        s = torch.sigmoid(x)
        if self.order == 1:
            return DenseLinear(torch.diag(s * (1 - s)))
        elif self.order == 2:
            n = self._in_dims[0]
            p = s * (1 - s) * (1 - 2 * s)
            eye_3 = torch.einsum('ij, jk -> ijk', torch.eye(n), torch.eye(n))
            diag_3_p = torch.einsum('ijk, k -> ijk', eye_3, p)
            return DenseSymmetricBilinear(diag_3_p)

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return FunctionComposition([self, other])
        else:
            return NotImplemented


# class ContractibleTensor:
#     """Including this operation implements * contraction"""
#
#     def __mul__(self, other):
#         """Contraction operation"""
#         assert isinstance(other, ContractibleTensor)
#
#         # assert isinstance(x, DenseVector)
#         # assert self.in_dims == x.out_dims
#         # in1_idx = self.all_idx()
#         # in2_idx = x.all_idx(offset=len(self.in_idx()))
#         # out_idx = self.out_idx()
#         # einsum_str = f"{in1_idx},{in2_idx}->{out_idx}"
#         # GLOBALS.p('doing einsum ', einsum_str)
#         # data = torch.einsum(einsum_str, self.data, x.data)
#         # return DenseVector(data)


# noinspection PyMissingConstructor
class OldLinearLayer(AtomicFunction):
    """Dense Linear Layer"""

    _out_dims: Tuple[int]
    _in_dims: Tuple[int]
    W: DenseLinear

    def __init__(self, W):
        W = to_pytorch(W)
        assert len(W.shape) == 2
        assert W.shape[0] >= 1
        assert W.shape[1] >= 1
        W = DenseLinear(W)
        self._out_dims = W.out_dims
        self._in_dims = W.in_dims
        self.W = W

    def d(self, order=1):
        return OldDLinearLayer(self.W, order=order)

    @property
    def out_dims(self) -> Tuple[int]:
        return self._out_dims

    @property
    def in_dims(self) -> Tuple[int]:
        return self._in_dims

    def __call__(self, x: Vector) -> DenseVector:
        assert isinstance(x, Vector)
        result = self.W * x
        assert isinstance(result, DenseVector)
        return result


class LinearLayer(AtomicFunction):
    """Dense Linear Layer"""

    W: TensorContraction

    def __init__(self, W, name=None):
        super().__init__(name=name)
        W = to_pytorch(W)
        self.W = TensorContraction.from_dense_matrix(W)

    def d(self, order=1):
        assert order >= 1
        if order == 1:
            # return ZeroFunction()
            return D_LinearLayer(self.W, order=order, base_name=self.human_readable)
        else:
            return ZeroFunction()

    def __call__(self, x: TensorContraction) -> TensorContraction:
        GLOBALS.increment_global_forward_flops(1)
        assert isinstance(x, TensorContraction)
        return self.W * x


class D_LinearLayer(AtomicFunction, LinearizedFunction):
    """derivative of Dense Linear Layer"""

    W: TensorContraction

    def in_dims(self):
        return self.W.in_dims

    def out_dims(self):
        return self.W.out_dims

    def __init__(self, W: TensorContraction, order=1, name=None, base_name=None):
        super().__init__(name=name, base_name=base_name, order=order)
        assert len(W.in_idx) == 1
        assert len(W.out_idx) == 1
        self.W = W

    def __call__(self, _unused_x: Tensor) -> TensorContraction:
        GLOBALS.increment_global_forward_flops(1)
        return self.W

    def d(self, order=1):
        assert order >= 1
        return ZeroFunction()


# noinspection PyMissingConstructor
class OldDLinearLayer(AtomicFunction, LinearizedFunction):
    """derivative of Dense Linear Layer"""

    W: DenseLinear

    def in_dims(self):
        return self.W.in_dims

    def out_dims(self):
        return self.W.out_dims

    def __init__(self, W: DenseLinear, order=1):
        # for now, only support matrices
        self.order = order
        assert len(W.in_idx()) == 1
        assert len(W.out_idx()) == 1
        self.W = W

    def __call__(self, _unused_x: Tensor) -> DenseLinear:
        return self.W

    def d(self, order=1):
        if order == 1:
            return self
        else:
            return Zero


class OldStructuredTensor(Tensor):
    # tensor in a structured form (einsum)
    # it supports lazy contraction with other tensors, calculating flop counts
    # performing the calculation

    tag: str  # tag helpful for debugging
    _in_dims: Tuple[int]
    _out_dims: Tuple[int]

    out_indices: List[chr]
    in_indices: List[chr]
    contracted_indices: List[chr]

    _index_spec_list: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
    _einsum_spec: str  # 'ij,jk->ik'

    tensors: List[torch.Tensor]  # [torch.ones((2,2,2)), torch.ones((2,2,2))]

    index_degree: Dict[chr, int]
    """for each index, count how many tensors share this index
    it is the degree of the "hyper-edge" labeled by this index in the Tensor Network Diagram
    d['i']==1 indicates a free index, i is a dangling edge
    d['i']==2 indicates contraction of two tensors, i is regular edge 
    d['i']==3 indicates contraction of three tensors, i is a hyper-edge connecting three tensors """

    # extra debugging, for each index, keep count of how many tensors have this index as out/in index
    # as well as the list of tensors
    index_out_degree: Dict[chr, int]
    index_in_degree: Dict[chr, int]
    index_out_tensors: Dict[chr, List[torch.Tensor]]  # d['i'] == [tensor1, tensor2, ...]
    index_in_tensors: Dict[chr, List[torch.Tensor]]

    index_dim: Dict[chr, int]
    "index dimensions, ie index_dim['i']==3"

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __init__(self, index_spec_list, tensors, tag=None):
        """['ij|k', 'k|lm'], [tensor1, tensor2]"""

        if tag is not None:
            self.tag = tag
        else:
            self.tag = f'tensor{GLOBALS.tensor_count:02d}'
            GLOBALS.tensor_count += 1

        index_spec_list = index_spec_list.copy()

        if len(index_spec_list) != len(tensors):
            GLOBALS.p(f"Provided {len(tensors)} tensors, but your index spec has {len(index_spec_list)} terms: ")
            for (i, term) in enumerate(index_spec_list):
                GLOBALS.p(f"term {i:2d}: {term:>20}")
                assert False

        self._index_spec_list = index_spec_list
        self.tensors = tensors
        self.index_degree = {}
        self.index_out_degree = {}
        self.index_in_degree = {}
        self.index_out_tensors = {}
        self.index_in_tensors = {}

        all_indices = set()  # all

        # create dict of sizes, by matching indices to tensors
        index_dim = {}  # ['ij'], [np.ones((2,5))] gives {'i': 2, 'j': 5}
        for (index_spec, tensor) in zip(index_spec_list, tensors):
            assert isinstance(index_spec, str)
            assert isinstance(tensor, torch.Tensor), f"Provided not an instance of torch.Tensor, {index_spec}, {tensor}"
            output_indices, input_indices = index_spec.split('|')
            all_indices_tensor = output_indices + input_indices

            # special handling for diagonal tensors
            if output_indices == input_indices:
                self.IS_DIAGONAL = True
            else:
                self.IS_DIAGONAL = False
                assert len(all_indices_tensor) == len(set(all_indices_tensor))
            if GLOBALS.PURE_TENSOR_NETWORKS:  # this disallows diagonal tensors
                assert not set(input_indices).intersection(set(output_indices))

            all_indices.update(set(all_indices_tensor))

            for idx in output_indices:
                # noinspection PyTypeChecker
                self.index_out_tensors.setdefault(idx, []).append(tensor)
                self.index_out_degree[idx] = self.index_out_degree.get(idx, 0) + 1
            for idx in input_indices:
                # noinspection PyTypeChecker
                self.index_in_tensors.setdefault(idx, []).append(tensor)
                self.index_in_degree[idx] = self.index_in_degree.get(idx, 0) + 1

            for idx in set(all_indices_tensor):
                self.index_degree[idx] = self.index_degree.get(idx, 0) + 1

            for (idx, dim) in zip(all_indices_tensor, tensor.shape):
                if idx in index_dim:
                    assert index_dim[idx] == dim, f"trying to set idx {idx} in indices {index_spec} to {dim}, " \
                                                  f"but it's already set to have dimension {index_dim[idx]}"
                assert dim > 0, f"Index {idx} has dimension {dim}"
                index_dim[idx] = dim
        self.index_dim = index_dim

        # sanity check, for each index make sure it appears equal number of times as contravariant and covariant
        self.contracted_indices = []
        self.out_indices = []
        self.in_indices = []
        for idx in sorted(all_indices):
            # number of tensors for which this idx is upper/contravariant
            out_count = len(self.index_out_tensors.get(idx, []))
            # number of tensors for which this idx is lower/covariant
            in_count = len(self.index_in_tensors.get(idx, []))
            assert out_count == self.index_out_degree.get(idx, 0)
            assert in_count == self.index_in_degree.get(idx, 0)

            if out_count and in_count:
                if not self.IS_DIAGONAL:
                    assert out_count == in_count
                if GLOBALS.PURE_TENSOR_NETWORKS:
                    assert out_count == 1  # in pure tensor networks, each index is contracted at most once
                else:
                    assert out_count <= 2, f"Index {idx} is contravariant in {out_count} tensors, suspicious," \
                                           f"it should be 1 for regular tensors, and 2 for diagonal matrices "
                assert idx not in self.contracted_indices, f"Trying to add {idx} as contracted index twice"
                self.contracted_indices.append(idx)
                self._check_indices_sorted()

            elif out_count and not in_count:
                assert idx not in self.out_indices, f"Trying to add {idx} as output index twice"
                self.out_indices.append(idx)
            elif in_count and not out_count:
                assert idx not in self.out_indices, f"Trying to add {idx} as input index twice"
                self.in_indices.append(idx)
            else:
                assert False, f"Shouldn't be here, {idx} is marked as occuring {out_count} times as contravariant " \
                              f"and {in_count} as covariant"

        assert len(self.out_indices) == len(set(self.out_indices))
        assert len(self.in_indices) == len(set(self.in_indices))
        assert not set(self.out_indices).intersection(self.in_indices)

        self._out_dims = tuple(self.index_dim[c] for c in self.out_indices)
        self._in_dims = tuple(self.index_dim[c] for c in self.in_indices)

        einsum_in = ','.join(index_spec.replace('|', '') for index_spec in self._index_spec_list)
        einsum_out = ''.join(self.out_indices) + ''.join(self.in_indices)

        if self.IS_DIAGONAL:
            self._einsum_spec = f'{input_indices}->{einsum_in}'
            self.in_indices = list(input_indices)
            self.out_indices = list(output_indices)
            # import pdb; pdb.set_trace()

        else:
            self._einsum_spec = f'{einsum_in}->{einsum_out}'

    @staticmethod
    def from_dense_vector(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object corresponding to given dense vector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord('i'), ord('i') + len(x.shape)))
        return OldStructuredTensor([idx + '|'], [x], tag)

    @staticmethod
    def from_dense_covector(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object corresponding to given dense covector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord('i'), ord('i') + len(x.shape)))
        return OldStructuredTensor(['|' + idx], [x], tag)

    @staticmethod
    def from_dense_matrix(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object (LinearMap with 1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        return OldStructuredTensor(['i|j'], [x], tag)

    @staticmethod
    def from_dense_linearmap(x: torch.Tensor, tag: str = None, idx: str = None):
        return OldStructuredTensor([idx], [x], tag)

    @staticmethod
    def from_diag_matrix(x: torch.Tensor, tag: str = None):
        """Creates StructuredTensor object (LinearMap with 1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        assert x.shape[0] > 0
        return OldStructuredTensor(['i|i'], [x], tag)

    # @staticmethod(x)

    def rename_index(self, old_name, new_name):
        # GLOBALS.p(f"naming {tag}:{old_name} to {new_name}")

        def rename_dictionary_entry(d: Dict[chr, Any], old_name: chr, new_name: chr):
            if old_name not in d:
                return
            assert isinstance(d, dict)
            assert new_name not in d
            assert isinstance(old_name, str)
            assert len(old_name) == 1
            assert isinstance(new_name, str)
            assert len(new_name) == 1
            d[new_name] = d[old_name]
            del d[old_name]

        def rename_list_entry(l, old_name, new_name):  # {len(l.count(old_name)}!=1
            if old_name not in l:
                return
            assert isinstance(l, list)
            assert isinstance(old_name, str)
            assert len(old_name) == 1
            assert isinstance(new_name, str)
            assert len(new_name) == 1

            # assert l.count(old_name) == 1, f"Found  {l.count(old_name)} instances of {old_name} in {l}"
            pos = l.index(old_name)
            l[pos] = new_name

        rename_list_entry(self.out_indices, old_name, new_name)
        rename_list_entry(self.in_indices, old_name, new_name)
        rename_list_entry(self.contracted_indices, old_name, new_name)
        # _index_spec_list: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
        for i, index_spec in enumerate(self._index_spec_list):
            self._index_spec_list[i] = index_spec.replace(old_name, new_name)
        #  _einsum_spec: str  # 'ij,jk->ik'
        self._einsum_spec = self._einsum_spec.replace(old_name, new_name)
        rename_dictionary_entry(self.index_degree, old_name, new_name)
        rename_dictionary_entry(self.index_out_degree, old_name, new_name)
        rename_dictionary_entry(self.index_in_degree, old_name, new_name)
        rename_dictionary_entry(self.index_out_tensors, old_name, new_name)
        rename_dictionary_entry(self.index_in_tensors, old_name, new_name)
        rename_dictionary_entry(self.index_dim, old_name, new_name)
        rename_list_entry(self.out_indices, old_name, new_name)
        rename_list_entry(self.in_indices, old_name, new_name)
        rename_list_entry(self.contracted_indices, old_name, new_name)

    def _check_indices_sorted(self):
        assert more_itertools.is_sorted(self.out_indices, strict=True)
        assert more_itertools.is_sorted(self.contracted_indices, strict=True)
        assert more_itertools.is_sorted(self.in_indices, strict=True)
        if not self.IS_DIAGONAL:
            assert more_itertools.is_sorted(self.out_indices + self.contracted_indices + self.in_indices, strict=True)

        # check that output/input indices are consecutive
        if self.out_indices:
            assert self.out_indices[-1] == chr(ord(self.out_indices[0]) + len(self.out_indices) - 1)
        # input indices won't be consecutive after partial contractions
        if self.in_indices:
            if not GLOBALS.ALLOW_PARTIAL_CONTRACTIONS:
                assert self.in_indices[-1] == chr(ord(self.in_indices[0]) + len(self.in_indices) - 1)

    def contract(self, other: 'OldStructuredTensor'):
        # GLOBALS.p('', self._index_spec_list)
        # GLOBALS.p('other old spec list', other._index_spec_list)

        # relabeling invariants
        # self.input_indices are larger than any other indices
        # other.contracted_indices + output_indices are larger than input indices
        #
        # increment all indices of "other" to make all k "other" input indices match first k self "output" indices
        # increment remaining self.output indices to be larger than largest input index.

        # is_sorted:

        self._check_indices_sorted()
        other._check_indices_sorted()

        left = OldStructuredTensor(self._index_spec_list, self.tensors)
        right = OldStructuredTensor(other._index_spec_list, other.tensors)

        # assert len(set(left.in_indices).union(right.out_indices)) > 0, "Outer products not supported"

        # first increment (never decrement because of _check_indices_sorted invariant) indices on the right to match
        # inputs
        assert len(right.out_indices) <= len(left.in_indices)

        # special handling for outer products
        if len(left.in_indices) == 0:
            incr1 = len(set(left.out_indices + left.contracted_indices))
        else:
            incr1 = ord(left.in_indices[0]) - ord(right.out_indices[0])
        if not GLOBALS.ALLOW_UNSORTED_INDICES:
            assert incr1 >= 0, f"Problem matching right tensor's {right.out_indices} to left tensor's {left.in_indices}, " \
                               f"we are assuming right tensors indices are incremented, never decremented"

        for idx in reversed(sorted(set(right.in_indices + right.out_indices + right.contracted_indices))):
            if incr1 > 0:
                right.rename_index(idx, chr(ord(idx) + incr1))

        # then increment uncontracted+output indices of the right to avoid interfering with indices on the left

        # actually maybe don't need to because left's contracted indices are strictly lower
        # incr2 = len(right.out_indices) - len(left.in_indices)
        # assert incr2 >= 0, f"Right tensor has more output indices {right.out_indices} than left has input indices " \
        #                   f"{left.in_indices}"
        # for idx in left.contracted_indices:

        # finally, increment uncontracted input indices on the left to avoid interfering with contracted/input indices
        # left.input_indices except left's contracted input indices incremented by
        for idx in set(left.in_indices).difference(right.out_indices):
            # move 1 beyond largest input index of the right tensor
            offset = int(ord(max(right.out_indices))) + 1 - int(ord(min(set(left.in_indices).difference(right.out_indices))))
            #  offset = len(set(right.contracted_indices+right.in_indices))
            if offset > 0:
                left.rename_index(idx, chr(ord(idx) + offset))

        # GLOBALS.p('my new spec list', left._index_spec_list)
        # GLOBALS.p('right new spec list', right._index_spec_list)

        result = OldStructuredTensor(left._index_spec_list + right._index_spec_list, left.tensors + right.tensors)
        GLOBALS.p(f'contracting {self.tag} and {other.tag}')
        GLOBALS.p(','.join(self._index_spec_list) + ' * ' + ','.join(other._index_spec_list) + ' = ' + ','.join(result._index_spec_list))
        return result

    def __mul__(self, other):
        return self.contract(other)

    @property
    def value(self):
        if self.IS_DIAGONAL:  # torch.einsum doesn't support 'i->ii' kind of einsum, do it manually
            assert len(self.in_indices) == 1, "Only support diagonal rank-2 tensors"
            assert len(self.tensors) == 1
            return torch.diag(self.tensors[0])

        # hack to deal with diagonal tensors
        ein_in, ein_out = self._einsum_spec.split('->')
        new_terms = []
        for term in ein_in.split(','):
            if len(term) == 2 and term[0] == term[1]:
                new_term = term[0]
            else:
                new_term = term
            new_terms.append(new_term)

        new_einsum_spec = ','.join(new_terms) + '->' + ein_out
        if new_einsum_spec != self._einsum_spec:
            GLOBALS.p("Warning, diagonal hack")
            return torch.einsum(new_einsum_spec, *self.tensors)
        return torch.einsum(self._einsum_spec, *self.tensors)

    @property
    def flops(self):
        """Flops required to materialize this tensor after einsum optimization"""

        views = oe.helpers.build_views(self._einsum_spec, self.index_dim)
        path, info = oe.contract_path(self._einsum_spec, *views, optimize='dp')
        return int(info.opt_cost)

    def _print_schedule(self):
        """Prints contraction schedule obtained by einsum optimizer"""

        einsum_str = self._einsum_spec
        sizes_dict = self.index_dim

        # indices: ['ij','jk','kl','lm']
        indices = einsum_str.split('->')[0].split(',')
        output_indices = einsum_str.split('->')[1]
        # unique_inds = set(einsum_str) - {',', '-', '>'}
        # index_size = [5]*len(unique_inds)
        # sizes_dict = dict(zip(unique_inds, index_size))
        views = oe.helpers.build_views(einsum_str, sizes_dict)

        # path: contraction path in einsum optimizer format, ie, [(0,), (2,), (1, 3), (0, 2), (0, 1)]
        path, info = oe.contract_path(einsum_str, *views, optimize='dp')

        # TODO(y): replace terms with something user provided
        # terms: ['term1', 'term2', 'term3', 'term4']
        terms = [f'term{i}' for i in range(len(indices))]
        GLOBALS.p('optimizing ', einsum_str, terms)
        GLOBALS.p('flops: ', info.opt_cost)

        # output_subscript: ['kl']
        output_subscript = output_indices

        input_index_sets = [set(x) for x in indices]
        output_indices = set(output_subscript)

        derived_count = 0
        for i, contract_inds in enumerate(path):
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            # GLOBALS.p(f'contracting {contract_inds}, input {input_index_sets}, output {output_indices}')
            contract_tuple = oe_helpers.find_contraction(contract_inds, input_index_sets, output_indices)
            out_inds, input_index_sets, _, idx_contract = contract_tuple
            # GLOBALS.p(f'idx_contract {idx_contract}, out_inds {out_inds}')

            current_input_index_sets = [indices.pop(x) for x in contract_inds]
            current_terms = [terms.pop(x) for x in contract_inds]

            # Last contraction
            if (i - len(path)) == -1:
                current_output_indices = output_subscript
                derived_term = f'derived{derived_count}'
            else:
                all_input_inds = "".join(current_input_index_sets)
                current_output_indices = "".join(sorted(out_inds, key=all_input_inds.find))
                derived_term = f'derived{derived_count}'
                derived_count += 1

            indices.append(current_output_indices)
            terms.append(derived_term)

            new_terms = []
            new_sets = []
            # for i in natsort.index_natsorted(current_terms):
            for i in natsort.index_natsorted(current_input_index_sets):
                new_terms.append(current_terms[i])
                new_sets.append(current_input_index_sets[i])
            # einsum_str = ",".join(current_input_index_sets) + "->" + current_output_indices
            #        GLOBALS.p(f'{derived_term}=einsum({einsum_str}, {current_terms})')
            einsum_str = ",".join(new_sets) + "->" + current_output_indices
            GLOBALS.p(f'{derived_term}=einsum({einsum_str}, {new_terms})')


class TensorContractionChain:
    """Represents contraction chain of multiple structured tensors. Keeps them in original form because one might
    call D on this

    D(dh1(f[1:]) * dh2(f[2:])) -> D(dh1(f[1:])) * dh2(f[2:]) + dh1(f[1:])*D(dh2(f[2:]))

    # supports ".flops" and ".value" fields which perform optimization/computation


    """

    children: List[OldStructuredTensor]

    @property
    def flops(self):
        assert 1 != 0, "not implemented"
        return 0

    @property
    def value(self):
        result = self.children[0]
        for c in self.children[1:]:
            result = result.contract(c)
        return result.value


class ZeroFunction(Function):
    def __call__(self, x):
        return ZeroTensor()

    @property
    def human_readable(self):
        return 'f_zero'


class IdentityFunction(Function):
    def __call__(self, t: 'Tensor'):
        return t

    @property
    def human_readable(self):
        return 'f_one'


def check_close(observed, truth, rtol=1e-5, atol=1e-8, label: str = '') -> None:
    """Convenience method for check_equal with tolerances defaulting to typical errors observed in neural network
    ops in float32 precision."""
    return check_equal(observed, truth, rtol=rtol, atol=atol, label=label)


def check_equal(observed, truth, rtol=1e-9, atol=1e-12, label: str = '') -> None:
    """
    Assert fail any entries in two arrays are not close to each to desired tolerance. See np.allclose for meaning of rtol, atol

    """

    if isinstance(truth, ZeroTensor):
        assert isinstance(observed, ZeroTensor) or observed == 0
        return
    if isinstance(observed, ZeroTensor):
        assert isinstance(truth, ZeroTensor) or truth == 0
        return

    if hasattr(observed, 'value'):
        observed = observed.value

    if hasattr(truth, 'value'):
        truth = truth.value

    # special handling for lists, which could contain
    # if type(observed) == List and type(truth) == List:
    #    for a, b in zip(observed, truth):
    #        check_equal(a, b)

    truth = to_numpy(truth)
    observed = to_numpy(observed)

    # broadcast to match shapes if necessary
    if observed.shape != truth.shape:
        #        common_shape = (np.zeros_like(observed) + np.zeros_like(truth)).shape
        truth = truth + np.zeros_like(observed)
        observed = observed + np.zeros_like(truth)

    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    # run np.testing.assert_allclose for extra info on discrepancies
    if not np.allclose(observed, truth, rtol=rtol, atol=atol, equal_nan=True):
        GLOBALS.p(f'Numerical testing failed for {label}')
        np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol, equal_nan=True)


def create_linear(mat):
    mat = to_pytorch(mat)
    d1, d2 = mat.shape
    layer = nn.Linear(d1, d2, bias=False)
    layer.weight.data = mat
    return layer


def to_pytorch(x) -> torch.Tensor:
    """Convert numeric object to floating point PyTorch tensor."""
    return from_numpy(to_numpy(x))


def to_numpy(x, dtype: np.dtype = None) -> np.ndarray:
    """
    Convert numeric object to floating point numpy array. If dtype is not specified, use PyTorch default dtype.

    Args:
        x: numeric object
        dtype: numpy dtype, must be floating point

    Returns:
        floating point numpy array
    """

    assert np.issubdtype(dtype, np.floating), "dtype must be real-valued floating point"

    # Convert to normal_form expression from a special form (https://reference.wolfram.com/language/ref/Normal.html)
    if hasattr(x, 'normal_form'):
        x = x.normal_form()

    if type(x) == np.ndarray:
        assert np.issubdtype(x.dtype, np.floating), f"numpy type promotion not implemented for {x.dtype}"

    if hasattr(x, "detach"):
        dtype = pytorch_dtype_to_floating_numpy_dtype(x.dtype)
        return x.detach().cpu().numpy().astype(dtype)

    # list or tuple, iterate inside to convert PyTorch arrrays
    if type(x) in [list, tuple]:
        x = [to_numpy(r) for r in x]

    # Some Python type, use numpy conversion
    result = np.array(x, dtype=dtype)
    assert np.issubdtype(result.dtype, np.number), f"Provided object ({result}) is not numeric, has type {result.dtype}"
    if dtype is None:
        return result.astype(pytorch_dtype_to_floating_numpy_dtype(torch.get_default_dtype()))
    return result


def pytorch_dtype_to_floating_numpy_dtype(dtype):
    """Converts PyTorch dtype to numpy floating point dtype, defaulting to np.float32 for non-floating point types."""
    if dtype == torch.float64:
        dtype = np.float64
    elif dtype == torch.float32:
        dtype = np.float32
    elif dtype == torch.float16:
        dtype = np.float16
    else:
        dtype = np.float32
    return dtype


def from_numpy(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.tensor(x)

import math
import sys

import pytest
import torch

def test_dense():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    # U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = OldLinearLayer(W0)
    x = DenseVector(x0)
    check_equal(W(x).value, W0 @ x0)

    dW = D(W)  # derivative of linear layer
    print(dW(ZeroTensor()) * x)  # get
    check_equal(dW(ZeroTensor()) * x, W0 @ x0)


def test_contract():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    W = DenseLinear(W0)
    U = DenseLinear(U0)
    x0 = to_pytorch([1, 2])
    x = DenseVector(x0)
    y = DenseCovector(x0)

    check_equal(W * U, W0 @ U0)
    assert isinstance(W * U, LinearMap)
    check_equal(W * x, [-3, 5])
    assert isinstance(W * x, Vector)
    check_equal(y * W, [-5, 6])
    assert isinstance(y * W, Covector)
    check_equal(y * x, 5)
    assert isinstance(y * x, Scalar)


def _old_create_unit_test_a():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = OldLinearLayer(W0)
    U = OldLinearLayer(U0)
    nonlin = OldRelu(x0.shape[0])
    loss = OldLeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss


def _create_unit_test_a():
    GLOBALS.reset_function_count()
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = LinearLayer(W0, name='W')
    U = LinearLayer(U0, name='U')
    nonlin = Relu(name='relu')
    loss = LeastSquares(name='lsqr')
    x = TensorContraction.from_dense_vector(x0, label='x')
    return W0, U0, x0, x, W, nonlin, U, loss

def _create_large_identity_network():
    GLOBALS.reset_function_count()
    d = 1000
    depth = 10
    W0 = torch.eye(d)

    layers = []
    for layer_num in range(depth):
        layers.append(LinearLayer(W0, name=f'W-{layer_num}'))

    # nonlin = Relu(name='relu')
    loss = LeastSquares(name='lsqr')
    x0 = torch.ones((d,))

    x = TensorContraction.from_dense_vector(x0, label='x')
    # noinspection PyTypeChecker
    return [loss] + layers, x


def _create_medium_network():
    GLOBALS.reset_function_count()
    torch.manual_seed(1)
    d = 1000
    depth = 10

    layers = []
    value_tensors = []
    for layer_num in range(depth):
        nonlin = Relu(name=f'relu-{layer_num}')
        layers.append(nonlin)
        W0 = torch.randn((d, d)) * math.sqrt(1/d)
        value_tensors.append(W0)
        layers.append(LinearLayer(W0, name=f'W-{layer_num}'))

    loss = LeastSquares(name='lsqr')
    x0 = torch.ones((d,))

    x = TensorContraction.from_dense_vector(x0, label='x')
    # noinspection PyTypeChecker
    return [loss] + layers, x, list(reversed(value_tensors))


def _create_medium_semirandom_network(skip_nonlin=False, width=2, depth=2):
    GLOBALS.reset_function_count()
    torch.manual_seed(1)

    layers = []
    value_tensors = []

    W0_ = to_pytorch([[1, -2], [-3, 4]])
    U0_ = to_pytorch([[5, -6], [-7, 8]])

    for layer_num in range(depth):
        if layer_num == 0 and width == 2:
            W0 = W0_
        elif layer_num == 1 and width == 2:
            W0 = U0_

        else:
            if layer_num == 0:
                W0 = torch.randn((width, width)) * math.sqrt(1 / width)
            else:
                W0 = torch.eye(width)

        layers.append(LinearLayer(W0, name=f'W-{layer_num}'))
        value_tensors.append(W0)
        nonlin = Relu(name=f'relu-{layer_num}')
        if layer_num < depth - 1 and not skip_nonlin:
            layers.append(nonlin)

    x0 = torch.ones((width,))
    if width >= 1:
        x0[1] = 2

    x = TensorContraction.from_dense_vector(x0, label='x')
    # noinspection PyTypeChecker
    return list(reversed(layers)), x, value_tensors


def _create_unit_test_a_sigmoid():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = OldLinearLayer(W0)
    U = OldLinearLayer(U0)
    nonlin = OldSigmoid(x0.shape[0])
    loss = OldLeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss


def test_unit_test_a():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    check_equal(dh1(a1), W0)
    check_equal(dh2(a2), [[0, 0], [0, 1]])
    check_equal(dh3(a3), [[5, -6], [-7, 8]])
    check_equal(dh4(a4), [-30, 40])
    check_equal(dh4(a4) * dh3(a3), [-430, 500])
    check_equal(dh4(a4) * dh3(a3) * dh2(a2), [0, 500])
    check_equal(dh4(a4) * dh3(a3) * dh2(a2) * dh1(a1), [-1500, 2000])

    GLOBALS.reset_global_forward_flops()
    assert GLOBALS.get_global_forward_flops() == 0

    result = f(x)
    check_equal(result, 1250)
    assert GLOBALS.get_global_forward_flops() == 4
    _unused_result = f(x)
    assert GLOBALS.get_global_forward_flops() == 4

    # creating new composition does not reuse cache
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    _unused_result = f(x)
    assert GLOBALS.get_global_forward_flops() == 2 * 4

    # partial composition test
    GLOBALS.reset_global_forward_flops()
    print('flops ', GLOBALS.get_global_forward_flops())
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    f._bind(x)
    # result = f(x)
    a2 = f[3:](x)  # input into h2
    assert GLOBALS.get_global_forward_flops() == 1
    check_equal(a2, [-3, 5])

    a4 = f[1:](x)  #
    assert GLOBALS.get_global_forward_flops() == 3
    check_equal(a4, [-30, 40])

    a5 = f[:](x)  #
    assert GLOBALS.get_global_forward_flops() == 4
    check_equal(a5, 1250)

    a5 = f[0:](x)  #
    assert GLOBALS.get_global_forward_flops() == 4
    check_equal(a5, 1250)


def test_sigmoid():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    _unused_a3 = h2(a2)

    nonlin = OldSigmoid(x0.shape[0])
    print('d sigmoid', D(nonlin)(a2))
    print('d2 sigmoid', dont_use_D2(nonlin)(a2))
    print(dont_use_D2(nonlin).order)

    check_close(nonlin(a2), [0.0474259, 0.993307])
    check_close(D(nonlin)(a2), [[0.0451767, 0], [0, 0.00664806]])
    check_close(dont_use_D2(nonlin)(a2), [[[0.0408916, 0], [0, 0]], [[0, 0], [0, -0.00655907]]])

    assert isinstance(dont_use_D2(nonlin)(a2), SymmetricBilinearMap)


def test_relu():
    f = OldRelu(2)
    df = f.d1  # also try D(f)
    # TODO(y): arguments to functions don't have Tensor semantics, so change type
    result = df(DenseVector([-3, 5]))
    check_equal(result, [[0, 0], [0, 1]])

    df = D(f)
    result = df(DenseVector([-3, 5]))
    check_equal(result, [[0, 0], [0, 1]])


def test_least_squares():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    _unused_a5 = h4(a4)

    assert isinstance(D(h4)(a4), Covector)
    assert isinstance(dont_use_D2(h4)(a4), QuadraticForm)
    check_equal(D(h4)(a4), a4)
    check_equal(dont_use_D2(h4)(a4), torch.eye(2))


def test_contraction():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == OldLinearLayer
    assert type(h4) == OldLeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)

    f(a1)  # run once to save activations

    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    _unused_deriv = dh1(f[1:]) * dh2(f[2:]) * dh3(f[3:])

    # TODO: change D functions to produce "structured tensor" objects
    # print(deriv.flops)  # prints the flop count
    # print(deriv.value)   # prints the value


def test_structured_tensor():
    d = 10
    x00 = torch.ones((d, d))
    y00 = torch.ones((d, d))
    z00 = 2 * torch.ones((d, d))

    a = OldStructuredTensor(['a|b', 'b|c'], [x00, y00])
    check_equal(a, x00 @ y00)
    assert a.flops == 2 * d ** 3

    x = OldStructuredTensor(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    check_equal(x, x00 @ y00 @ z00)

    x = OldStructuredTensor(['a|b'], [x00])
    y = OldStructuredTensor(['a|b'], [y00])
    z = OldStructuredTensor(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    x.contract(y)
    xyz = x * y * z
    assert xyz.flops == 4 * d ** 3
    check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = OldStructuredTensor.from_dense_vector(x00, 'col')
    row = OldStructuredTensor.from_dense_covector(x00, 'row')
    mat = OldStructuredTensor.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product

    check_equal(row * mat * mat * mat,
                x00 @ ma0 @ ma0 @ ma0)
    check_equal(mat * mat * mat * col,
                ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    check_equal(mat * mat * col * row * mat * mat,
                ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = OldStructuredTensor.from_diag_matrix(3 * x00, 'diag')
    dia0 = diag.value
    print(dia0)

    assert (row * mat * diag * mat).flops == 410  # structured reverse mode

    print()
    check_equal(row * mat * diag * mat,
                x00 @ ma0 @ dia0 @ ma0)

    # 3 x 2 grid example from "3 decompositions" section of
    # https://notability.com/n/wNU5UXNGENsmRBzMFDSJQ
    d = 2
    rank2 = torch.ones((d, d))
    rank3 = torch.ones((d, d, d))
    A = OldStructuredTensor.from_dense_covector(rank2, idx='ij', tag='A')
    B = OldStructuredTensor.from_dense_linearmap(rank3, idx='i|ml', tag='B')
    # C = StructuredTensor.from_dense_linearmap(rank2, idx='l|o', tag='C')
    # D = StructuredTensor.from_dense_linearmap(rank2, idx='j|k', tag='D')
    # E = StructuredTensor.from_dense_linearmap(rank3, idx='km|n', tag='E')
    # F = StructuredTensor.from_dense_vector(rank2, idx='no', tag='F')
    print(A * B)
    # K = A * B
    # K = A * B * C * D * E * F
    # disable some error checks
    # gl.ALLOW_PARTIAL_CONTRACTIONS = True
    # gl.ALLOW_UNSORTED_INDICES = True
    # K = A * B * C * D
    # TODO(y): non-determinism (probably because using set)
    #    print(K.value)
    #    print(K.flops)


def test_contractible_tensor2():
    d = 10
    x00 = torch.ones((d, d))
    y00 = torch.ones((d, d))
    z00 = 2 * torch.ones((d, d))

    diag = TensorContraction.from_diag_matrix(3 * torch.ones((3,)), label='diag')
    assert diag.out_idx == diag.in_idx
    assert len(diag.out_idx) == 1
    dia0 = diag.value
    print(dia0)

    a = TensorContraction([('a|b', x00), ('b|c', y00)])
    check_equal(a.value, x00 @ y00)
    assert a.flops == 2 * d ** 3

    x = TensorContraction.__legacy_init__(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    check_equal(x, x00 @ y00 @ z00)

    x = TensorContraction.__legacy_init__(['a|b'], [x00])
    y = TensorContraction.__legacy_init__(['a|b'], [y00])
    z = TensorContraction.__legacy_init__(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    a = TensorContraction.__legacy_init__(['a|b'], [x00])
    b = TensorContraction.__legacy_init__(['a|b'], [x00])
    print(a)
    c = a * b
    assert c.children_specs == ['a|b', 'b|c']
    check_equal(c, x00 @ x00)

    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))
    a = TensorContraction.__legacy_init__(['a|bc', 'bc|de'], [rank3, rank4], label='a')
    b = TensorContraction.__legacy_init__(['ab|c', 'c|d'], [rank3, rank2], label='b')
    c = a * b
    assert c.children_specs == ['a|bc', 'bc|de', 'de|f', 'f|g']

    a = TensorContraction.__legacy_init__(['a|bc', 'bc|defg'], [rank3, rank6], label='a')
    b = TensorContraction.__legacy_init__(['ab|c', 'c|d'], [rank3, rank2], label='b')
    c = a * b
    assert c.children_specs == ['a|bc', 'bc|defg', 'de|h', 'h|i']

    a = TensorContraction.__legacy_init__(['a|'], [rank1])
    b = TensorContraction.__legacy_init__(['|a'], [rank1])
    c = a * b
    assert c.children_specs == ['a|', '|b']

    check_equal(c, torch.outer(rank1, rank1))

    xyz = x * y * z
    assert xyz.flops == 4 * d ** 3
    check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = TensorContraction.from_dense_vector(x00, 'col')
    row = TensorContraction.from_dense_covector(x00, 'row')
    mat = TensorContraction.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product

    check_equal(row * mat * mat * mat,
                x00 @ ma0 @ ma0 @ ma0)
    check_equal(mat * mat * mat * col,
                ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    check_equal(mat * mat * col * row * mat * mat,
                ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = TensorContraction.from_diag_matrix(3 * x00, 'diag')
    assert diag.out_idx == diag.in_idx
    assert len(diag.out_idx) == 1
    dia0 = diag.value
    print(dia0)

    d2 = 2
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))

    # UnitTestB:
    A = torch.ones((2, 3, 2, 2))
    B = torch.ones((2,))
    C = torch.ones((2, 2, 2, 4))

    D = TensorContraction([('ab|cd', A, 'A'), ('c|c', B, 'B'), ('cd|ef', C, 'C')])
    assert D.ricci_str == 'ab|cd,c|c,cd|ef->ab|ef'

    a = TensorContraction.__legacy_init__(['a|bc', 'bc|de'], [rank3, rank4], label='a')
    b = TensorContraction.__legacy_init__(['ab|c', 'c|d'], [rank3, rank2], label='b')
    c = a * b

    check_equal(row * diag, x00 @ dia0)
    check_equal(row * mat * diag, x00 @ ma0 @ dia0)

    result = row * mat
    print(result.ricci_str)
    assert result.ricci_str == '|a,a|b->|b'
    assert (row * mat * diag).ricci_str == '|a,a|b,b|b->|b'
    assert (row * mat * diag).ricci_str == '|a,a|b,b|b->|b'
    assert (row * mat * diag * mat).flops == 410  # structured reverse mode

    print()
    check_equal(row * mat * diag * mat,
                x00 @ ma0 @ dia0 @ ma0)

    # 2x3 grid example from "3 decompositions" section of
    # https://notability.com/n/wNU5UXNGENsmRBzMFDSJQ
    d = 2
    rank2 = torch.ones((d, d))
    rank3 = torch.ones((d, d, d))

    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))

    A = ('|ij', rank2, 'A')
    B = ('i|lm', rank3, 'B')
    C = ('l|o', rank2, 'C')
    D = ('j|k', rank2, 'D')
    E = ('km|n', rank3, 'E')
    F = ('no|', rank2, 'F')
    K = TensorContraction([A, B, C, D, E, F])
    assert K.flops == 104, "Change in flop requirement detected (don't actually know if 104 is correct)"


@pytest.mark.skip(
    reason="this example doesn't work because our implementation currently contracts sequentially left to right only with automatic index renaming")
def test_2x3grid_mul():
    """If we want this kind of contraction to work, need to redo contraction logic to look at particular names of indices and match
    them up. Current logic just matches positions and discards original index names: all k output indices of the right are renamed to match first k input indices
    of the left."""

    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))

    A = TensorContraction.from_dense_covector(rank2, idx='ij', label='A')
    B = TensorContraction.from_dense_tensor('i|lm', rank3, 'B')
    C = TensorContraction.from_dense_tensor('l|o', rank2, 'C')
    D = TensorContraction.from_dense_tensor('j|k', rank2, 'D')
    E = TensorContraction.from_dense_tensor('km|n', rank3, 'E')
    F = TensorContraction.from_dense_vector(rank2, idx='no', label='F')

    print('=====' * 20)
    partial1 = A * B
    print('-----', partial1.ricci_str)
    assert partial1.ricci_out == '|jlm'
    partial2 = partial1 * C
    print('-----', partial2.ricci_str)
    assert partial1.ricci_out == '|jlm'
    partial3 = partial2 * D
    print('-----', partial3.ricci_str)

    # at this point error happens, D^j is contracted with B_l instead of A_j
    assert partial1.ricci_out == '|lmo'
    partial4 = partial3 * E
    assert partial4.ricci_out == '|mok'
    print('-----', partial2.ricci_str)
    partial5 = partial4 * F
    print('-----', partial2.ricci_str)

    K = A * B * C * D * E * F
    print(K.flops)
    print(K._einsum_spec)

    # Hessian-like contractions

    # this should raise an error because rank doesn't match
    with pytest.raises(Exception):
        A = TensorContraction([('a|bc', rank2, 'A')])

    assert A.ricci_str == 'a|bc'
    B = TensorContraction([('bc|defg', rank6, 'B')])
    AB = A * B
    print(AB.ricci_str)


def test_partial_contraction_UnitTestC():
    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))

    # this should raise an error because rank doesn't match
    with pytest.raises(Exception):
        A = TensorContraction([('a|bc', rank2, 'A')])

    A = TensorContraction([('a|bc', rank3, 'A')])
    assert A.ricci_str == 'a|bc->a|bc'
    B = TensorContraction([('bc|defg', rank6, 'B')])
    assert B.ricci_str == 'bc|defg->bc|defg'
    AB = A * B
    assert AB.ricci_str == 'a|bc,bc|defg->a|defg'
    result2 = AB * TensorContraction([('de|h', rank3, 'C'), ('h|i', rank2, 'D')])
    result1 = TensorContraction([('a|bc', rank3, 'A'), ('bc|defg', rank6, 'B'), ('de|h', rank3, 'C'), ('h|i', rank2, 'D')])
    print(result1.ricci_str)
    assert result2.ricci_out == 'a|ifg'

    # UnitTestD
    vec = TensorContraction.from_dense_vector(rank1)
    assert (result2 * vec).ricci_out == 'a|fg'
    assert (result2 * vec * vec).ricci_out == 'a|g'
    assert (result2 * vec * vec * vec).ricci_out == 'a|'


"""
def test_overall():
    # reverse mode

    from einograd import jacobian, forward, to_expression

    W = create_linear([[1, -2], [-3, 4]])
    U = create_linear([[5, -6], [-7, 8]])
    loss_func = LeastSquaresLoss()
    out = to_expression(nn.Sequential(W, U, U, loss_func))
    data = to_pytorch([1, 2])
    x = data
    d = jacobian(out, x)
    print(d.flops)  # 600 FLOPs required for optimized backward
    forward(out, x)  # runs the forward pass, saves input to every layer
    print(d.value)  # materialize the actual value

    # forward mode
    out = torch.Sequential[col, W, W, W]
    d = jacobian(out, x)
    print(d.flops)  # 600
    forward(net, x)
    print(d.value)  # materialize the actual value

    out = torch.Sequential([W, relu, W, xent])
    d = jacobian(out, x)
    print(d.flops)  # 410
    forward(net, x)
    print(d.value)  # materialize the actual value

    # cross-country mode
    net = torch.Sequential([W, W, row, col, W, W])
    d = jacobian(net, x)
    print(d.flops)  # 1000

    loss = torch.Sequential([W, relu, W, relu, xent])
    x = W.weight
    vector = Tensor(torch.ones(x.shape))
    hess = jacobian(jacobian(loss, x), x)
    hvp = hess @ vector
    print(hvp.flops)
    print(hvp.value)
"""
def test_names():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    assert W.human_readable == 'W'
    assert (W @ U).human_readable == '(W@U)'
    GLOBALS.reset_function_count()
    new_layer0 = LinearLayer(W0)
    new_layer1 = LinearLayer(W0)
    assert new_layer0.human_readable == 'LinearLayer'
    assert new_layer1.human_readable == 'LinearLayer01'
    assert (new_layer0 * new_layer1).human_readable == '(LinearLayer*LinearLayer01)'

    GLOBALS.reset_function_count()
    dW = D(W)
    assert dW.base_name == 'W'
    assert dW.human_readable == 'D_W'
    assert dW.human_readable == 'D_W'
    assert D(D(W)).human_readable == 'f_zero'

    loss1 = LeastSquares()
    dloss1 = D(loss1)
    dloss2 = D(loss1)
    assert loss1.human_readable == 'LeastSquares'
    assert dloss1.human_readable == 'D_LeastSquares'
    assert dloss2.human_readable == 'D_LeastSquares01'

    ddloss1 = D(D(loss1))
    assert ddloss1.human_readable == 'D_D_LeastSquares'
    assert id(GLOBALS.function_dict['D_D_LeastSquares']) == id(ddloss1)


def test_derivatives():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    f_slow = UnmemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == LinearLayer
    assert type(h4) == LeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    check_equal(dh1(a1), W0)
    check_equal(dh2(a2), [[0, 0], [0, 1]])
    check_equal(dh3(a3), [[5, -6], [-7, 8]])

    # simple hessian
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    D_W = D(W)
    assert D_W.human_readable == 'D_W'
    D_D_W = D(D_W)
    assert D_D_W.human_readable == 'f_zero'

    def hess(f):
        return D(D(f))

    print(hess(W)(x))
    check_equal(hess(W)(x), 0)

    # loss._bind(x)
    check_equal(hess(loss)(x), torch.eye(2))
    func = loss @ W
    func._bind(x)
    first_deriv = D(func)
    # first_deriv._bind(x)
    second_deriv = D(first_deriv)
    print(second_deriv)
    print(hess(U))

    print(hess(loss @ W))
    print(D(loss @ W))

    # check end-to-end derivative
    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    GLOBALS.reset_function_count()

    f_slow = h1
    deriv = D(f_slow)
    check_equal(deriv(x), [[1., -2.], [-3., 4.]])

    f_slow = UnmemoizedFunctionComposition([h2, h1])
    deriv = D(f_slow)
    check_equal(deriv(x), [[0., 0.], [-3., 4.]])

    f_slow = UnmemoizedFunctionComposition([h3, h2, h1])
    deriv = D(f_slow)
    check_equal(deriv(x), [[18., -24.], [-24., 32.]])

    # f_slow = h4
    # deriv = D(f_slow)
    check_equal(D(h4)(a4), [-30, 40])

    f_slow = h4
    deriv = D(f_slow)
    check_equal(deriv(a4), [-30, 40])

    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    relu = h2
    lsqr = h4

    check_equal(D(W)(x), [[1., -2.], [-3., 4.]])
    check_equal((D(relu) @ W * D(W))(x), [[0., 0.], [-3., 4.]])
    check_equal((((D(U) @ relu @ W) * (D(relu)) @ W) * D(W))(x), [[18., -24.], [-24., 32.]])
    # ((D(h4) @ U @ relu @ W) * (D_U01 @ relu @ W) * (D_relu02 @ W) * D_W03)

    check_equal(D(h4)(a4), [-30, 40])
    check_equal(D(h3)(a3), [[5., -6.], [-7., 8.]])
    check_equal(D(h4)(a4) * D(h3)(a3), [-430, 500])

    dH4 = D(h4) @ U @ relu @ W
    dH3 = D(U) @ relu @ W
    dH2 = D(relu) @ W
    dH1 = D(W)
    check_equal((dH4 * dH3 * dH2 * dH1)(x), [-1500, 2000])

    f_slow = UnmemoizedFunctionComposition([h4, h3, h2, h1])
    deriv = D(f_slow)
    check_equal(deriv(x), [-1500, 2000])

    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = True
    W0 = torch.tensor([[1., -2.], [-3., 4.]])
    ii = torch.eye(2)
    check_equal(torch.einsum('ab,ac,bd->cd', ii, W0, W0), W0.T @ W0)
    check_equal([[10., -14.], [-14., 20.]], W0.T @ W0)
    # second derivatives
    GLOBALS.reset_function_count()
    f = lsqr @ W
    deriv = D(f)
    hess = D(deriv)
    check_equal(hess(x), [[10., -14.], [-14., 20.]])

    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False

    def hessian(f):
        return D(D(f))

    g = h3 @ h2 @ h1
    dg = D(g)
    check_equal(dg(x), [[18., -24.], [-24., 32.]])
    d2f = hessian(h4)
    check_equal(d2f(x), [[1, 0], [0, 1]])
    check_equal((d2f(x) * dg(x)) * dg(x), [[900., -1200.], [-1200., 1600.]])

    gauss_newton = ((d2f @ g) * dg) * dg
    check_equal(gauss_newton(x), [[900., -1200.], [-1200., 1600.]])

    f_slow = loss @ relu
    f_hess = hessian(f_slow)
    check_equal(f_hess(x), [[1, 0], [0, 1]])

    f_slow = loss @ U @ relu
    f_hess = hessian(f_slow)
    check_equal(f_hess(x), [[74., -86.], [-86., 100.]])

    GLOBALS.reset_function_count()
    g = h3 @ h2 @ h1
    dg = D(g)
    d2f = hessian(h4)
    gauss_newton = ((d2f @ g) * dg) * dg
    check_equal(gauss_newton(x), [[900., -1200.], [-1200., 1600.]])

    print(gauss_newton.human_readable)
    GLOBALS.reset_function_count()

    # this calculation only works with left-to-right composition order
    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    # GLOBALS.switch_composition_order = False

    f_slow = h4 @ (h3 @ h2 @ h1)
    myhess = hessian(f_slow)
    print(myhess.human_readable)

    print(hessian(f_slow)(x) * x)
    check_equal(hessian(f_slow)(x) * x, [-1500, 2000])

    check_equal(hessian(f_slow)(x).diag, [900, 1600])
    check_equal(hessian(f_slow)(x).trace, 2500)

    check_equal(hessian(f_slow)(x), [[900., -1200.], [-1200., 1600.]])
    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    # GLOBALS.switch_composition_order = True


def test_memoized_hvp():
    # test Hessian vector product against PyTorch implementation
    GLOBALS.reset_global_state()
    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False

    GLOBALS.enable_memoization = True

    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    # f = FunctionComposition([h4, h3, h2, h1])

    f = make_function_composition([h4, h3, h2, h1])
    f._bind(x)

    def hessian(f):
        return D(D(f))

    check_equal(hessian(f)(x) * x, [-1500, 2000])

    # obtain it using PyTorch
    from torch.autograd import Variable
    from torch import autograd

    class LeastSquaresLoss(nn.Module):
        def __init__(self):
            super(LeastSquaresLoss, self).__init__()
            return

        def forward(self, data, targets=None):
            if targets is None:
                targets = torch.zeros_like(data)

            if len(data.shape) == 1:
                err = data - targets
            else:
                err = data - targets.view(-1, data.shape[1])
            return torch.sum(err * err) / 2

    def hvp(loss, param, v):
        grad_f, = autograd.grad(loss, param, create_graph=True)
        z = grad_f.flatten() @ v
        hvp, = autograd.grad(z, param, retain_graph=True)
        # hvp, = autograd.grad(grad_f, param, v.view_as(grad_f))  # faster versio 531 -> 456
        return hvp

    b = 1

    W = create_linear([[1, -2], [-3, 4]])
    U = create_linear([[5, -6], [-7, 8]])
    loss = LeastSquaresLoss()

    print("\nrelu")
    nonlin = nn.ReLU()
    layers = [W, nonlin, U]

    net = nn.Sequential(*layers)

    x0 = to_pytorch([1, 2])
    x_var = Variable(x0, requires_grad=True)
    loss0 = loss(net(x_var))
    check_equal(hvp(loss0, x_var, x0), hessian(f)(x) * x)
    GLOBALS.reset_global_state()



def test_transpose():
    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False

    d = 10
    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = TensorContraction.from_dense_vector(x00, 'col')
    row = TensorContraction.from_dense_covector(x00, 'row')
    mat = TensorContraction.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product

    check_equal(row * mat * mat * mat,
                x00 @ ma0 @ ma0 @ ma0)
    check_equal(mat * mat * mat * col,
                ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    check_equal(mat * mat * col * row * mat * mat,
                ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = TensorContraction.from_diag_matrix(3 * x00, 'diag')
    assert diag.out_idx == diag.in_idx
    assert len(diag.out_idx) == 1
    dia0 = diag.value
    print(dia0)

    d2 = 2
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))

    # UnitTestB:
    A = torch.ones((2, 3, 2, 2))
    B = torch.ones((2,))
    C = torch.ones((2, 2, 2, 4))

    D = TensorContraction([('ab|cd', A, 'A'), ('c|c', B, 'B'), ('cd|ef', C, 'C')])
    assert D.ricci_str == 'ab|cd,c|c,cd|ef->ab|ef'

    a = TensorContraction.__legacy_init__(['a|bc', 'bc|de'], [rank3, rank4], label='a')
    b = TensorContraction.__legacy_init__(['ab|c', 'c|d'], [rank3, rank2], label='b')
    c = a * b

    check_equal(row * diag, x00 @ dia0)
    check_equal(row * mat * diag, x00 @ ma0 @ dia0)

    result = row * mat
    print(result.ricci_str)
    assert result.ricci_str == '|a,a|b->|b'
    assert (row * mat * diag).ricci_str == '|a,a|b,b|b->|b'
    assert (row * mat * diag).ricci_str == '|a,a|b,b|b->|b'

    assert (row * mat * diag).T.ricci_str == 'b|b,b|a,a|->b|'  # have b|b,a|b,|a->|b

    A0 = torch.tensor([[1, 2], [3, 4]])
    A = TensorContraction.from_dense_matrix(A0)
    At0 = A.T.value
    check_equal(A0.T, At0)


# Tests from "Diagonal logic"
def test_diagonal_problem():
    d = 2
    m = 20
    torch.manual_seed(1)
    row0 = torch.randint(1, m, size=(d,)).float()
    col0 = torch.randint(1, m, size=(d,)).float()
    diag0 = torch.randint(1, m, size=(d,)).float()
    ma0 = torch.randint(1, m, size=(d, d)).float()
    row = TensorContraction.from_dense_covector(row0, label='row')
    col = TensorContraction.from_dense_vector(col0, label='col')
    diag = TensorContraction.from_diag_matrix(diag0, label='diag')
    mat = TensorContraction.from_dense_matrix(ma0, label='mat')

    assert (row * diag).ricci_str == '|a,a|a->|a'
    check_equal(row * diag, row0 * diag0)

    assert (diag * col).ricci_str == 'a|a,a|->a|'
    check_equal(diag * col, diag0 * col0)

    assert (row * col).ricci_str == '|a,a|->|'
    check_equal(row * col, row0 @ col0)

    # weighted dot product
    assert (row * diag * col).ricci_str == '|a,a|a,a|->|'
    check_equal(row * diag * col, (row0 * diag0 * col0).sum())

    # Hadamard product of two diagonal matrices support combining, but not direct materialization for now, need to figure out how to deal
    # with multiple diagonal matrices, only support 1
    assert (diag * diag).ricci_str == 'a|a,a|a->a|a'
    with pytest.raises(Exception):
        print((diag * diag).value)
        check_equal(diag * diag, torch.diag(diag0) @ torch.diag(diag0))

    # this case could be enabled in the future, but to reduce scope currently
    # we specialize all contractions to go in left-to-right-order
    with pytest.raises(Exception):
        assert (col * diag).ricci_str == 'a|,a|a->a|'

    check_equal(mat * diag, ma0 @ torch.diag(diag0))
    check_equal(diag * mat, torch.diag(diag0) @ ma0)


def test_diagonal_and_trace():
    A = TensorContraction([('|ab', from_numpy([[1, 2], [3, 4]]), 'A')])
    r = A.diag
    assert r.ricci_out == '|a'
    check_equal(A.diag, [1, 4])
    check_equal(A.trace, 5)

    # matrices are treated as linear forms, so no trace defined, this should raise error
    with pytest.raises(Exception):
        A = TensorContraction.from_dense_matrix(from_numpy([[1, 2], [3, 4]]))
        print(A.trace)


def test_nesting():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    f = loss @ U @ nonlin @ W
    print(f.human_readable)


def test_outer_product():
    d = 10
    x00 = torch.ones((d, d))
    y00 = torch.ones((d, d))
    z00 = 2 * torch.ones((d, d))

    a = TensorContraction.__legacy_init__(['a|b', 'b|c'], [x00, y00])
    check_equal(a, x00 @ y00)
    assert a.flops == 2 * d ** 3

    x = TensorContraction.__legacy_init__(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    check_equal(x, x00 @ y00 @ z00)

    x = TensorContraction.__legacy_init__(['a|b'], [x00])
    y = TensorContraction.__legacy_init__(['a|b'], [y00])
    z = TensorContraction.__legacy_init__(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    xyz = x * y * z
    assert xyz.flops == 4 * d ** 3
    check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = TensorContraction.from_dense_vector(x00, 'col')
    row = TensorContraction.from_dense_covector(x00, 'row')
    mat = TensorContraction.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product


def test_factored_diagonal():
    d = 10
    x00 = torch.ones((d,))
    B = TensorContraction([('|a', x00), ('|b', x00)])
    assert B.diag.flops == 10
    assert B.flops == 100


def test_derivatives_factored():
    GLOBALS.reset_global_state()
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    f_slow = UnmemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == LinearLayer
    assert type(h4) == LeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    check_equal(dh1(a1), W0)
    check_equal(dh2(a2), [[0, 0], [0, 1]])
    check_equal(dh3(a3), [[5, -6], [-7, 8]])

    # simple hessian
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    D_W = D(W)
    assert D_W.human_readable == 'D_W'
    D_D_W = D(D_W)
    assert D_D_W.human_readable == 'f_zero'

    def hess(f):
        return D(D(f))

    print(hess(W)(x))
    check_equal(hess(W)(x), 0)

    check_equal(hess(loss)(x), torch.eye(2))
    first_deriv = D(loss @ W)
    second_deriv = D(first_deriv)
    print(second_deriv)
    print(hess(U))

    print(hess(loss @ W))
    print(D(loss @ W))

    # check end-to-end derivative
    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    GLOBALS.reset_function_count()

    f_slow = h1
    deriv = D(f_slow)
    check_equal(deriv(x), [[1., -2.], [-3., 4.]])

    f_slow = make_function_composition([h2, h1])
    deriv = D(f_slow)
    check_equal(deriv(x), [[0., 0.], [-3., 4.]])

    f_slow = make_function_composition([h3, h2, h1])
    deriv = D(f_slow)
    check_equal(deriv(x), [[18., -24.], [-24., 32.]])

    # f_slow = h4
    # deriv = D(f_slow)
    check_equal(D(h4)(a4), [-30, 40])

    f_slow = h4
    deriv = D(f_slow)
    check_equal(deriv(a4), [-30, 40])

    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    relu = h2
    lsqr = h4

    check_equal(D(W)(x), [[1., -2.], [-3., 4.]])
    check_equal((D(relu) @ W * D(W))(x), [[0., 0.], [-3., 4.]])
    check_equal((((D(U) @ relu @ W) * (D(relu)) @ W) * D(W))(x), [[18., -24.], [-24., 32.]])
    # ((D(h4) @ U @ relu @ W) * (D_U01 @ relu @ W) * (D_relu02 @ W) * D_W03)

    check_equal(D(h4)(a4), [-30, 40])
    check_equal(D(h3)(a3), [[5., -6.], [-7., 8.]])
    check_equal(D(h4)(a4) * D(h3)(a3), [-430, 500])

    dH4 = D(h4) @ U @ relu @ W
    dH3 = D(U) @ relu @ W
    dH2 = D(relu) @ W
    dH1 = D(W)
    check_equal((dH4 * dH3 * dH2 * dH1)(x), [-1500, 2000])

    f_slow = make_function_composition([h4, h3, h2, h1])
    deriv = D(f_slow)
    check_equal(deriv(x), [-1500, 2000])

    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = True
    W0 = torch.tensor([[1., -2.], [-3., 4.]])
    ii = torch.eye(2)
    check_equal(torch.einsum('ab,ac,bd->cd', ii, W0, W0), W0.T @ W0)
    check_equal([[10., -14.], [-14., 20.]], W0.T @ W0)
    # second derivatives
    GLOBALS.reset_function_count()
    f = lsqr @ W
    deriv = D(f)
    hess = D(deriv)
    check_equal(hess(x), [[10., -14.], [-14., 20.]])

    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False

    def hessian(f):
        return D(D(f))

    g = h3 @ h2 @ h1
    dg = D(g)
    check_equal(dg(x), [[18., -24.], [-24., 32.]])
    d2f = hessian(h4)
    check_equal(d2f(x), [[1, 0], [0, 1]])
    check_equal((d2f(x) * dg(x)) * dg(x), [[900., -1200.], [-1200., 1600.]])

    gauss_newton = ((d2f @ g) * dg) * dg
    check_equal(gauss_newton(x), [[900., -1200.], [-1200., 1600.]])

    f_slow = loss @ relu
    f_hess = hessian(f_slow)
    check_equal(f_hess(x), [[1, 0], [0, 1]])

    f_slow = loss @ U @ relu
    f_hess = hessian(f_slow)
    check_equal(f_hess(x), [[74., -86.], [-86., 100.]])

    GLOBALS.reset_function_count()
    g = h3 @ h2 @ h1
    dg = D(g)
    d2f = hessian(h4)
    gauss_newton = ((d2f @ g) * dg) * dg
    check_equal(gauss_newton(x), [[900., -1200.], [-1200., 1600.]])

    print(gauss_newton.human_readable)
    GLOBALS.reset_function_count()

    # this calculation only works with left-to-right composition order
    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    # GLOBALS.switch_composition_order = False

    f_slow = h4 @ (h3 @ h2 @ h1)
    myhess = hessian(f_slow)
    print(myhess.human_readable)

    print(hessian(f_slow)(x) * x)
    check_equal(hessian(f_slow)(x) * x, [-1500, 2000])

    check_equal(hessian(f_slow)(x).diag, [900, 1600])
    check_equal(hessian(f_slow)(x).trace, 2500)

    check_equal(hessian(f_slow)(x), [[900., -1200.], [-1200., 1600.]])
    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    # GLOBALS.switch_composition_order = True

    # GLOBALS.DEBUG_HESSIANS = False
    # myhess = hessian(f_slow)
    myhess = hessian(f_slow)
    diag_flops_regular = diag(myhess(x)).flops

    GLOBALS.FULL_HESSIAN = False
    myhess = hessian(f_slow)
    diag_flops_factored = diag(myhess(x)).flops
    full_flops_factored = myhess(x).flops

    print(diag_flops_regular, diag_flops_factored, full_flops_factored)
    assert diag_flops_regular == 64, "Change detected"
    assert diag_flops_factored == 38, "Change detected"

    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    GLOBALS.FULL_HESSIAN = True


def test_activation_reuse():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    GLOBALS.enable_memoization = True
    f = make_function_composition([h4, h3, h2, h1])

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    check_equal(dh1(a1), W0)
    check_equal(dh2(a2), [[0, 0], [0, 1]])
    check_equal(dh3(a3), [[5, -6], [-7, 8]])
    check_equal(dh4(a4), [-30, 40])
    check_equal(dh4(a4) * dh3(a3), [-430, 500])
    check_equal(dh4(a4) * dh3(a3) * dh2(a2), [0, 500])
    check_equal(dh4(a4) * dh3(a3) * dh2(a2) * dh1(a1), [-1500, 2000])

    GLOBALS.reset_global_forward_flops()
    assert GLOBALS.get_global_forward_flops() == 0

    result = f(x)
    check_equal(result, 1250)
    assert GLOBALS.get_global_forward_flops() == 4
    _unused_result = f(x)
    assert GLOBALS.get_global_forward_flops() == 4

    # creating new composition does not reuse cache
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = make_function_composition([h4, h3, h2, h1])
    _unused_result = f(x)
    assert GLOBALS.get_global_forward_flops() == 2 * 4

    # partial composition test
    GLOBALS.reset_global_forward_flops()
    print('flops ', GLOBALS.get_global_forward_flops())
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    f._bind(x)
    # result = f(x)
    a2 = f[3:](x)  # input into h2
    assert GLOBALS.get_global_forward_flops() == 1
    check_equal(a2, [-3, 5])

    a4 = f[1:](x)  #
    assert GLOBALS.get_global_forward_flops() == 3
    check_equal(a4, [-30, 40])

    a5 = f[:](x)  #
    assert GLOBALS.get_global_forward_flops() == 4
    check_equal(a5, 1250)

    a5 = f[0:](x)  #
    assert GLOBALS.get_global_forward_flops() == 4
    check_equal(a5, 1250)

    GLOBALS.reset_global_forward_flops()
    f._bind(x)
    result = f(x)  # this "bind" x to the composition, all partial computations now reuse this value
    check_equal(result, 1250)

    check_equal(dh1(a1), [[1, -2], [-3, 4]])
    g = D(h1)
    check_equal(g(x), [[1, -2], [-3, 4]])

    check_equal(dh2(a2) * dh1(a1), [[0, 0], [-3, 4]])
    g = (D(h2) @ f[3:]) * D(h1)
    check_equal(g(x), [[0, 0], [-3, 4]])

    check_equal(dh3(a3) * dh2(a2) * dh1(a1), [[18, -24], [-24, 32]])
    g = (D(h3) @ f[2:]) * (D(h2) @ f[3:]) * D(h1)
    check_equal(g(x), [[18, -24], [-24, 32]])

    check_equal(dh3(a3) * dh2(a2) * dh1(a1), [[18, -24], [-24, 32]])
    g = (D(h4) @ f[1:]) * (D(h3) @ f[2:]) * (D(h2) @ f[3:]) * D(h1)
    check_equal(g(x), [-1500, 2000])

    check_equal(D(f)(x), [-1500, 2000])

    GLOBALS.reset_global_forward_flops()
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    f._bind(x)
    result = f(x)  # this "bind" x to the composition, all partial computations now reuse this value
    check_equal(result, 1250)
    assert GLOBALS.get_global_forward_flops() == 4

    # should reuse previous values, no increase in forward call count
    result2 = f[1:](x)
    check_equal(result2, [-30, 40])
    assert GLOBALS.get_global_forward_flops() == 4

    GLOBALS.enable_memoization = False


def test_activation_reuse2():
    GLOBALS.reset_global_forward_flops()
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    f._bind(x)
    # result = f(x)  # this "bind" x to the composition, all partial computations now reuse this value
    # check_equal(result, 1250)
    g = D(f)
    check_equal(g(x), [-1500, 2000])
    # (D(h4) @ f[1:]) * (D(h3) @ f[2:]) * (D(h2) @ f[3:]) * D(h1)
    assert GLOBALS.get_global_forward_flops() == 7  # 4 derivatives, and 3 forward activations
    check_equal(g(x), [-1500, 2000])

def test_hvp():
    # test Hessian vector product against PyTorch implementation
    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False

    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    # f = FunctionComposition([h4, h3, h2, h1])

    f = make_function_composition([h4, h3, h2, h1])

    def hessian(f):
        return D(D(f))

    check_equal(hessian(f)(x) * x, [-1500, 2000])

    # obtain it using PyTorch
    from torch.autograd import Variable
    from torch import autograd

    class LeastSquaresLoss(nn.Module):
        def __init__(self):
            super(LeastSquaresLoss, self).__init__()
            return

        def forward(self, data, targets=None):
            if targets is None:
                targets = torch.zeros_like(data)

            if len(data.shape) == 1:
                err = data - targets
            else:
                err = data - targets.view(-1, data.shape[1])
            return torch.sum(err * err) / 2

    def hvp(loss, param, v):
        grad_f, = autograd.grad(loss, param, create_graph=True)
        z = grad_f.flatten() @ v
        hvp, = autograd.grad(z, param, retain_graph=True)
        # hvp, = autograd.grad(grad_f, param, v.view_as(grad_f))  # faster versio 531 -> 456
        return hvp

    b = 1

    W = create_linear([[1, -2], [-3, 4]])
    U = create_linear([[5, -6], [-7, 8]])
    loss = LeastSquaresLoss()

    print("\nrelu")
    nonlin = nn.ReLU()
    layers = [W, nonlin, U]

    net = nn.Sequential(*layers)

    x0 = to_pytorch([1, 2])
    x_var = Variable(x0, requires_grad=True)
    loss0 = loss(net(x_var))
    check_equal(hvp(loss0, x_var, x0), hessian(f)(x) * x)
    print("Expected HVP is ", (hessian(f)(x) * x).value)


def test_medium_hvp():
    # test Hessian vector product against PyTorch implementation
    GLOBALS.enable_memoization = True
    GLOBALS.reset_global_state()

    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    skip_nonlin = False
    layers, x, value_tensors = _create_medium_semirandom_network(width=2, depth=2, skip_nonlin=skip_nonlin)

    # Working version
    # GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = True
    # skip_nonlin = True
    # layers, x, value_tensors = _create_medium_semirandom_network(width=2, depth=1, skip_nonlin=skip_nonlin)


    # Working version
    #    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = True
    #    skip_nonlin = False
    #    layers, x, value_tensors = _create_medium_semirandom_network(width=2, depth=1, skip_nonlin=skip_nonlin)

    # this one doesn't work
    #    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = True
    #    skip_nonlin = False
    #    layers, x, value_tensors = _create_medium_semirandom_network(width=2, depth=2, skip_nonlin=skip_nonlin)

    g = make_function_composition(layers)
    f = LeastSquares(name='lsqr') @ g
    f._bind(x)

    # print(value_tensors)

    # check_equal(f[1:](x), [-3, 5])
    # check_equal(f[3:](x), [1, 2])   # this returns [-1, 1] instead of [1, 2]

    def hessian(f):
        return D(D(f))

    g = D(f)
    h = hessian(f)
    # print("Hessian Trace efficient: ", trace(h(x)).flops/10**9)

    print("Gradient Flops: ", g(x).flops)
    print("HVP Flops: ", (h(x) * x).flops)
    print("Hessian Flops: ", h(x).flops)
    print("Hessian Trace: ", trace(h(x)).flops)

    hvp_ours = (h(x) * x)
    hvp_ours0 = hvp_ours.value
    if skip_nonlin:
        if len(value_tensors) == 2 and not skip_nonlin:
            if value_tensors[0].shape == (2, 2):
                check_equal(hvp_ours0, [-2926., 4336.])

    if len(value_tensors) == 2 and not skip_nonlin:
        if value_tensors[0].shape == (2, 2):
            check_equal(hvp_ours0, [-1500, 2000])

    hess_ours = h(x)
    hess_ours0 = hess_ours.value
    if len(value_tensors) == 1 and skip_nonlin and value_tensors[0].shape[0] == 2:
        print(hess_ours0)  # should be {{10., -14.}, {-14., 20.}}
        check_equal(hvp_ours0, [-18., 26.])



    # obtain it using PyTorch
    from torch.autograd import Variable
    from torch import autograd

    class LeastSquaresLoss(nn.Module):
        def __init__(self):
            super(LeastSquaresLoss, self).__init__()
            return

        def forward(self, data, targets=None):
            if targets is None:
                targets = torch.zeros_like(data)

            if len(data.shape) == 1:
                err = data - targets
            else:
                err = data - targets.view(-1, data.shape[1])
            return torch.sum(err * err) / 2

    def hvp(loss, param, v):
        grad_f, = autograd.grad(loss, param, create_graph=True)
        z = grad_f.flatten() @ v
        hvp, = autograd.grad(z, param, retain_graph=True)
        # hvp, = autograd.grad(grad_f, param, v.view_as(grad_f))  # faster versio 531 -> 456
        return hvp

    pytorch_loss = LeastSquaresLoss()
    pytorch_layers = []
    for (i, value_tensor) in enumerate(value_tensors):
        pytorch_layers.append(create_linear(value_tensor))
        nonlin = nn.ReLU()
        if i < len(value_tensors) - 1 and not skip_nonlin:
            pytorch_layers.append(nonlin)

    net = nn.Sequential(*pytorch_layers)
    x0 = x.value

    x_var = Variable(x0, requires_grad=True)
    loss0 = pytorch_loss(net(x_var))
    hvp_theirs = hvp(loss0, x_var, x0)

    print("difference is ", torch.norm(hvp_ours0 - hvp_theirs))
    check_equal(hvp_theirs, hvp_ours0, rtol=1e-5, atol=1e-5)
    GLOBALS.reset_global_state()

@pytest.mark.skip()
def test_larger_factored_hessian():
    GLOBALS.reset_global_state()
    GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False
    GLOBALS.enable_memoization = True

    layers, x, value_tensors = _create_medium_network()
    f = make_function_composition(layers)
    f._bind(x)

    def hessian(f):
        return D(D(f))

    h = hessian(f)

    g = D(f)
    h = hessian(f)
    print("Gradient Flops: ", g(x).flops/10**9)
    print("HVP Flops: ", (h(x) * x).flops/10**9)
    print("Hessian Flops: ", h(x).flops/10**9)
    print("Hessian Trace: ", trace(h(x)).flops/10**9)

    GLOBALS.FULL_HESSIAN = False
    h = hessian(f)
    print("Hessian Trace efficient: ", trace(h(x)).flops/10**9)


def test_thu():
    GLOBALS.reset_global_state()
    GLOBALS.function_dict = {}
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Relu()
    loss = LeastSquares()
    x = TensorContraction.from_dense_vector(x0, label='x')

    g = U @ nonlin @ W
    objective = loss @ g
    hessian = D(D(objective))
    hvp = hessian(x) * x
    print(hvp.flops)    # 44 flops
    print(hvp.value)    # [-1500.,  2000]
    print(D(objective)(x).flops)   # 26 flops

    dg = D(g)
    d2l = D(D(loss))
    gauss_newton = ((d2l @ g) * dg) * dg
    print('gn: ', gauss_newton(x).flops)    # 72 flops
    print('hessian: ', hessian(x).flops)    # 72 flops

    print('diff: ', torch.norm(hessian(x).value-gauss_newton(x).value))  # tensor(0.)

    layers, x, value_tensors = _create_medium_network()
    f = make_function_composition(layers)

    g = D(f)
    h = D(D(f))
    print("Gradient Flops: ", g(x).flops/10**9)
    print("HVP Flops: ", (h(x) * x).flops/10**9)
    print("Hessian Flops: ", h(x).flops/10**9)
    print("Hessian Diagonal: ", diag(h(x)).flops/10**9)
    print("Hessian Trace: ", trace(h(x)).flops/10**9)

    GLOBALS.FULL_HESSIAN = False
    h = D(D(f))
    print("Hessian Trace efficient: ", trace(h(x)).flops/10**9)
    print("Hessian Diag efficient: ", diag(h(x)).flops/10**9)


test_activation_reuse()
test_activation_reuse2()
test_contract()
test_contractible_tensor2()
test_contraction()
test_dense()
test_derivatives()
test_derivatives_factored()
test_diagonal_and_trace()
test_diagonal_problem()
test_factored_diagonal()
test_hvp()
test_larger_factored_hessian()
test_least_squares()
test_medium_hvp()
test_memoized_hvp()
test_names()
test_nesting()
test_outer_product()
test_partial_contraction_UnitTestC()
test_relu()
test_sigmoid()
test_structured_tensor()
test_thu()
test_transpose()
test_unit_test_a()
