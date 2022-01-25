"""Base types used everywhere"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from typing import Union

import more_itertools
import natsort
import numpy as np
import opt_einsum as oe
import torch
from opt_einsum import helpers as oe_helpers
from torch import nn as nn

import util as u


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

    def __init__(self):
        self.DEBUG = True
        self.device = 'cpu'
        self.PURE_TENSOR_NETWORKS = False
        self.tensor_count = 0
        self.function_count = {}  # {LinearLayer: 1, Relu: 2}
        self.function_dict = {}  # {LinearLayer: 1, Relu: 2}
        self.ALLOW_PARTIAL_CONTRACTIONS = True  # allow some indices of the left tensor to remain uncontracted
        self.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES = False   # the original order turned out to be wrong for the Hessian case
        self.ALLOW_UNSORTED_INDICES = False
        self.MAX_INDEX_COUNT = 1000
        self.idx0 = 'a'
        self.all_indices = set(chr(ord(self.idx0) + i) for i in range(self.MAX_INDEX_COUNT))
        self.all_indices_list = tuple(chr(ord(self.idx0) + i) for i in range(self.MAX_INDEX_COUNT))

    def generate_tensor_name(self):
        name = f'T{self.tensor_count:02d}'
        self.tensor_count += 1
        return name

    def reset_function_count(self):
        self.function_count = {}
        self.function_dict = {}

    def reset_tensor_count(self):
        self.tensor_count = 0


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


#    @abstractmethod
#    def in_dims(self):
#        pass

#    @abstractmethod
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
        print('========')
        print(in_idx_order)
        # for each in index determine the largest index tensor for which it's an in-index
        in_idx_to_rightmost_tensor = {}
        for (idx, tensor_id_tuple) in in_idx_order:
            for tensor_id in tensor_id_tuple:
                in_idx_to_rightmost_tensor[idx] = max(in_idx_to_rightmost_tensor.get(idx, -1), tensor_id)

        print('in_idx_order ', in_idx_order)
        # print('index i has rank', in_idx_to_rightmost_tensor['i'])
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
                print("diagonal tensor, no einsum for " + ','.join(self.children_specs))
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
        print('optimizing ', einsum_str, terms)
        print('flops: ', info.opt_cost)

        # output_subscript: ['kl']
        output_subscript = output_indices

        input_index_sets = [set(x) for x in indices]
        output_indices = frozenset(output_subscript)

        derived_count = 0
        for i, contract_inds in enumerate(path):
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            # print(f'contracting {contract_inds}, input {input_index_sets}, output {output_indices}')
            contract_tuple = oe_helpers.find_contraction(contract_inds, input_index_sets, output_indices)
            out_inds, input_index_sets, _, idx_contract = contract_tuple
            # print(f'idx_contract {idx_contract}, out_inds {out_inds}')

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
            #        print(f'{derived_term}=einsum({einsum_str}, {current_terms})')
            einsum_str = ",".join(new_sets) + "->" + current_output_indices
            print(f'{derived_term}=einsum({einsum_str}, {new_terms})')

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

        print(f'renaming {old_name} to {new_name}')

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
        for (spec, data, name) in self._original_specs:
            out_idx, in_idx = spec.split('|')
            new_spec = in_idx+'|'+out_idx
            new_specs.insert(0, (new_spec, data, name))
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
        print(f'contracting {self} with {other}')
        #        self._check_indices_sorted()
        #        other._check_indices_sorted()

        # left = ContractibleTensor2(self._original_specs, tag=f'temp_{self.label}')
        left = self

        # clone the right tensor to make use of index relabeling without corrupting given right tensor
        right = TensorContraction(list(other._original_specs), label=f'cloned_{other.label}')

        # this can potentially be relaxed to make contractions commutative
        # however, all current applications only need left-to-right order, and this was the only case that's well tested, hence require it
        if not GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES:
            assert len(left.in_idx) >= len(right.out_idx), f"only allow partial contraction on left tensor, right tensor must contract all output indices, contracting {left.ricci_out} with {right.ricci_out}"

        # rename indices of right to avoid clashes
        if len(right.contracted_idx + right.in_idx):
            print('before step 1 rename')
            print(right.children_specs)

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
        print('before step 2 rename', left.children_specs, right.children_specs)
        if not GLOBALS.CHANGE_DEFAULT_ORDER_OF_FINDING_IN_INDICES:
            left_contracted = left.in_idx[:len(right.out_idx)]  # contract these in-indices of LEFT with all out-indices of RIGHT
        else:
            # find in_idx corresponding to highest-rank tensor

            in_idx_ranks = {}
            in_idx_rank_tensors = {}
            for idx in left.in_idx:
                print('doing index ', idx)
                if len(left.idx_to_in_tensors[idx]) == 1:
                    _tensor_idx = left.idx_to_in_tensors[idx][0]
                    _tensor = self.children_data[_tensor_idx]
                    in_idx_ranks.setdefault(len(_tensor.shape), []).append(idx)
                    # ensure we don't try to add twice
                    if len(_tensor.shape) in in_idx_rank_tensors:
                        assert in_idx_rank_tensors[len(_tensor.shape)] == _tensor_idx
                    else:
                        in_idx_rank_tensors[len(_tensor.shape)] = _tensor_idx
            if not in_idx_ranks.keys():
                # fall back on previous method, which works for diagonal tensors
                left_contracted = left.in_idx[:len(right.out_idx)]
            else:
                top_rank = max(in_idx_ranks.keys())
                assert len(in_idx_ranks[top_rank]) >= len(right.out_idx), "Couldn't find tensor to contract with right"
                left_contracted = in_idx_ranks[top_rank][:len(right.out_idx)]

        print(f'matching left {left_contracted} to right {right.out_idx}')
        for left_idx, right_idx in zip(left_contracted, right.out_idx):
            right._rename_index(right_idx, left_idx)
        print('after step 2 rename')
        print(right.children_specs)

        # TODO: here (add ..)
        new_specs = self._transpose_specs(left.children_specs + right.children_specs, left.children_data + right.children_data,
                                          left.children_labels + right.children_labels)
        result = TensorContraction(new_specs, label=f"{self.label}*{other.label}")
        print(f'contracting {self.label} and {other.label}')
        print(','.join(self.children_specs) + ' * ' + ','.join(other.children_specs) + ' = ' + ','.join(result.children_specs))
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
                print("Warning, diagonal hack")
                return torch.einsum(new_einsum_spec, *self.children_data)
        return torch.einsum(self._einsum_spec, *self.children_data)


class ZeroTensor(Tensor):
    """Arbitrary shape tensor of zeros"""

    def in_dims(self):
        return ()

    def out_dims(self):
        return ()


zero = ZeroTensor()
Zero = ZeroTensor()  # TODO(y) remove one of the zeros


class Scalar(Tensor, ABC):
    """Scalar, empty output and input indices"""

    def out_dims(self):
        return ()

    def in_dims(self):
        return ()


class Vector(Tensor, ABC):
    """pure contravariant Tensor, upper (output) indices only"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass


class Covector(Tensor, ABC):
    """pure covariant Tensor, lower (input) indices only"""

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


class LinearMap(Tensor, ABC):
    """mixed Tensor, one set of upper and one set of lower"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


class QuadraticForm(Tensor, ABC):
    # this must return ()
    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


class SymmetricBilinearMap(Tensor, ABC):
    """Symmetric bilinear map. Two sets of input indices with equal dimensions.
    TODO(y): enforce this somehow"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
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
    order: int    # derivative order

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
        if name not in ['@', '+', '*']:   # don't save automatically created composite functions
            assert name not in GLOBALS.function_dict, f"Function {name} has already been created"
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
        assert isinstance(other, Function)

        if isinstance(self, FunctionComposition):
            return FunctionComposition(self.children + [other])
        else:
            return FunctionComposition([self, other])

    @property
    def human_readable(self):
        """Return human-readable representation of f.

        ie, DLeastSquares@LinearLayer, (DDLeastSquares@LinearLayer)*DLinearLayer

        """

        if isinstance(self, CompositeFunction):
            child_str = self.name.join(str(c) for c in self.children)
            assert len(self.children) > 1, "found composition with one child or less"
            return f'({child_str})'  # (1+2)
        else:
            return self.name

    def __str__(self):
        return self.human_readable

    def __repr__(self):
        return str(self)


class Function(ABC):
    """Differentiable function"""

    @abstractmethod
    def __call__(self, t: 'Tensor'):
        pass

    # @abstractmethod
    # def in_dims(self):
    #    """Input (lower) dimensions"""
    #    pass

    # @abstractmethod
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


# TODO(y): rename to FunctionNesting? compositvefunction/functioncomposition clash
class FunctionComposition(CompositeFunction):
    def __init__(self, children: List['Function']):
        super().__init__(name='@')
        # self.name = '@'
        assert len(children) >= 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[-1](t)
        for c in reversed(self.children[:-1]):
            result = c(result)
        return result


def make_function_composition(children):
    if len(children) == 0:
        return ZeroFunction()
    elif isinstance(children[0], ZeroFunction):
        return ZeroFunction
    elif len(children) == 1:
        return children[0]
    else:
        return FunctionComposition(children)


class AtomicFunction(Function, FunctionSharedImpl):
    @property
    def d1(self):
        return self.d(1)

    @abstractmethod
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

    def __init__(self, order=1):
        assert order >= 1
        self.order = order

    # operator composition
    def __matmul__(self, other):
        if isinstance(other, D_):
            return D_(self.order + other.order)
        elif isinstance(other, Operator):
            assert False, "We don't have any other operators implemented"
        else:
            return NotImplemented

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
                print("differentiating ", c1, " got ", D(c1))
                for (j, c2) in enumerate(other.children):
                    if i == j:
                        mul_children.append(dc1)
                    else:
                        mul_children.append(c2)
                add_children.append(make_function_contraction(mul_children))
            return make_function_addition(add_children)

        # chain rule
        elif isinstance(other, FunctionComposition):
            mul_children = []
            for (i, c1) in enumerate(other.children):
                mul_children.append(make_function_composition([D(c1)] + other.children[i + 1:]))
            return make_function_contraction(mul_children)

        else:
            assert False, f"Unknown node type: {other}"


D = D_(order=1)
D2 = D @ D


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
        # print('other is ', other)
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
        return DenseScalar((x * x).sum() / 2)

    def d(self, order=1):
        return D_LeastSquares(order=order, base_name=self.human_readable)

    @property
    def in_dims(self):
        return -43

    @property
    def out_dims(self):
        return -45

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return FunctionComposition([self, other])
        else:
            return NotImplemented


class D_LeastSquares(AtomicFunction, LinearizedFunction):
    """Derivatives of LeastSquares"""

    def __init__(self, name=None, base_name=None, order: int = 1):
        super().__init__(name=name, base_name=base_name, order=order)

    def __call__(self, x: TensorContraction) -> TensorContraction:
        assert self.order <= 2, "third and higher order derivatives not implemented"

        if self.order == 1:
            return x.T
        elif self.order == 2:
            assert len(x.out_dims) == 1
            return TensorContraction.from_dense_quadratic_form(torch.eye(x.out_dims[0]))

    @property
    def d1(self):
        return self.d(1)

    def d(self, order=1):
        return D_LeastSquares(order=self.order + order)

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return FunctionComposition([self, other])
        else:
            return NotImplemented

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


# TODO(y): rename to IdentityLayer (to disambig from IdentityLinearMap)
# noinspection PyMissingConstructor
class OldIdentity(AtomicFunction):
    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = (dim,)

    def __call__(self, x: Tensor):
        return x

    def d(self, order=1):
        return OldDIdentity(self._in_dims[0])

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
class OldDIdentity(AtomicFunction):
    """Derivatives of identity"""

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

    @property
    def d1(self):
        return self.d(1)

    def d(self, order=1):
        if order == 1:
            return IdentityLinearMap()
        elif order >= 2:
            return Zero

    def __call__(self, x: DenseVector):
        assert self.order <= 2, "third and higher order derivatives not implemented"
        n = self._in_dims[0]

        if self.order == 1:
            return DenseLinear(torch.eye(n))
        elif self.order == 2:
            return 0

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
        return Zero

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

    @property
    def out_dims(self):
        return -42

    @property
    def in_dims(self):
        return -42

    def d(self, order=1):
        return Zero

    def __call__(self, x: TensorContraction) -> TensorContraction:
        if self.order == 1:
            x = x.value
            return TensorContraction.from_diag_matrix((x > torch.tensor(0)).float())

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return FunctionComposition([self, other])
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
#         # print('doing einsum ', einsum_str)
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


#    TODO(y): maybe also implement Function interface?
class MemoizedFunctionComposition:
    """Represents a composition of functions with memoization
    Unbound call, ie f@g@h, can be used as intermediate result for constructing a composition
    Bound call, ie (f@g@h)(x), at this point it is frozen and can't be modified.
    """

    children: List[Any]  # List[Function] fix: forward references
    # parent               # FunctionComposition type, multiple Compositions point here. fix: forward reference
    arg: Any

    def __init__(self, children, parent=None):
        self.arg = None
        self.parent = parent
        self.children = children
        self._saved_outputs = [None] * (len(children) + 1)

        # if creating a sub-composition, extra sanity check that the nodes we are using
        # are already pointing to the parent composition
        for child in children:
            if parent:
                assert child.parent == parent
            else:
                if hasattr(child, 'parent') and child.parent is not None:
                    assert False, f"Warning, Node {child} already has parent {child.parent}"
                child.parent = self

    def __matmul__(self, other):
        assert self.arg is None, "Can't combine compositions with bound parameters"
        if isinstance(other, Function):
            return MemoizedFunctionComposition(self.children + [other])
        else:
            return NotImplemented

    # only allow simple slicing
    def __getitem__(self, s):
        if isinstance(s, slice):
            if isinstance(s.step, int):
                assert s.step == 1
            error_msg = "this case hasn't been tested, for now only single level of parent  redirection is allowed"
            assert self.parent is None, error_msg
            backlink = self if self.parent is None else self.parent
            assert s.stop is None
            assert len(self.children[s.start:]) > 0, f"only have {len(self.children)} members of composition, " \
                                                     f"attempted to start at {s.start} "
            return MemoizedFunctionComposition(self.children[s.start:], backlink)
        else:
            assert False, "use [:] slicing as [i] is ambiguous"
            # assert isinstance(s, int)
            # return self.children[s]

    def _bind(self, arg):
        print('binding ', arg)
        self.arg = arg
        self._saved_outputs[len(self.children)] = arg

    def memoized_compute(self, node):
        """Composition([h3,h2,h1]): memoized_compute(h2) computes everything up to h2"""
        assert self.arg is not None, "arg not bound, call _bind first"
        assert id(self._saved_outputs[len(self.children)]) == id(self.arg)
        assert node in self.children, "Trying to compute {node} but it's not in Composition"
        idx = self.children.index(node)

        # last_saved gives position of function whose output has been cached
        # we treat "x" as a dummy function which returns itself, so
        # last_cached == len(children) means just the output of "x" was cached
        for last_cached in range(len(self.children)):
            if self._saved_outputs[last_cached] is not None:
                break
        else:
            last_cached = len(self.children)

        print(f'found saved output of {last_cached} node')
        for i in range(last_cached - 1, idx - 1, -1):
            if i == len(self.children):
                assert id(self._saved_outputs[last_cached]) == id(self.arg)
                continue

            u.increment_global_forward_flops(1)
            result = self.children[i](self._saved_outputs[i + 1])
            self._saved_outputs[i] = result
            print('saving output of ', i)

        return self._saved_outputs[idx]

    def __call__(self, arg: Vector) -> Vector:
        assert isinstance(arg, Vector), "must call function with Vector type"
        if self.parent is not None:
            assert isinstance(self.parent, MemoizedFunctionComposition)
            if self.parent.arg is not None:
                assert id(arg) == id(self.parent.arg)
            else:
                self.parent._bind(arg)
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
            print(f"Provided {len(tensors)} tensors, but your index spec has {len(index_spec_list)} terms: ")
            for (i, term) in enumerate(index_spec_list):
                print(f"term {i:2d}: {term:>20}")
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
        # print(f"naming {tag}:{old_name} to {new_name}")

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
        # print('', self._index_spec_list)
        # print('other old spec list', other._index_spec_list)

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

        # print('my new spec list', left._index_spec_list)
        # print('right new spec list', right._index_spec_list)

        result = OldStructuredTensor(left._index_spec_list + right._index_spec_list, left.tensors + right.tensors)
        print(f'contracting {self.tag} and {other.tag}')
        print(','.join(self._index_spec_list) + ' * ' + ','.join(other._index_spec_list) + ' = ' + ','.join(result._index_spec_list))
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
            print("Warning, diagonal hack")
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
        print('optimizing ', einsum_str, terms)
        print('flops: ', info.opt_cost)

        # output_subscript: ['kl']
        output_subscript = output_indices

        input_index_sets = [set(x) for x in indices]
        output_indices = set(output_subscript)

        derived_count = 0
        for i, contract_inds in enumerate(path):
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            # print(f'contracting {contract_inds}, input {input_index_sets}, output {output_indices}')
            contract_tuple = oe_helpers.find_contraction(contract_inds, input_index_sets, output_indices)
            out_inds, input_index_sets, _, idx_contract = contract_tuple
            # print(f'idx_contract {idx_contract}, out_inds {out_inds}')

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
            #        print(f'{derived_term}=einsum({einsum_str}, {current_terms})')
            einsum_str = ",".join(new_sets) + "->" + current_output_indices
            print(f'{derived_term}=einsum({einsum_str}, {new_terms})')


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
        print(f'Numerical testing failed for {label}')
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
