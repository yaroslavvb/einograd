"""Base types used everywhere"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

import more_itertools
import natsort
import opt_einsum as oe
import torch
from opt_einsum import helpers as oe_helpers

import util as u



# GLOBALS = AttrDict({'DEBUG': True, 'device': 'cpu', 'PURE_TENSOR_NETWORKS': False,
#               'tensor_count': 0, 'ALLOW_PARTIAL_CONTRACTIONS': False,
#               'ALLOW_UNSORTED_INDICES': False,
#               'MAX_INDEX_COUNT': 1000, 'idx0': 'a'})

class _GLOBALS_CLASS:
    __shared_state = {}

    def __init__(self):
        self.DEBUG = True
        self.device = 'cpu'
        self.PURE_TENSOR_NETWORKS = False
        self.tensor_count = 0
        self.ALLOW_PARTIAL_CONTRACTIONS = True  # allow some indices of the left tensor to remain uncontracted
        self.ALLOW_UNSORTED_INDICES = False
        self.MAX_INDEX_COUNT = 1000
        self.idx0 = 'a'
        self.all_indices = set(chr(ord(self.idx0) + i) for i in range(self.MAX_INDEX_COUNT))
        self.all_indices_list = tuple(chr(ord(self.idx0) + i) for i in range(self.MAX_INDEX_COUNT))

    def generate_tensor_name(self):
        name = f'T{self.tensor_count:02d}'
        self.tensor_count += 1
        return name


GLOBALS = _GLOBALS_CLASS()


##################################################
# Tensors
##################################################


class TensorSharedImpl(ABC):
    # addition
    def __add__(self, other: 'Tensor'):
        assert isinstance(other, Tensor)
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

    @abstractmethod
    def in_dims(self):
        pass

    @abstractmethod
    def out_dims(self):
        pass


class AtomicTensor(Tensor, ABC):
    pass


class CompositeTensor(Tensor, TensorSharedImpl, ABC):
    children: List[Tensor]

    # TODO(y) drop dimensions? These are are only needed at tensor level
    def out_dims(self):
        pass

    def in_dims(self):
        pass


class TensorAddition(CompositeTensor):
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



class TensorContraction(Tensor):
    label: str  # tag helpful for debugging

    # see UnitTestB for illustration
    _original_specs: Tuple[Tuple]  # [('ab|cd', A, 'A'), ...]
    tensor_specs: List[str]  # [ab|cd, c|c, cd|ef], has to be a list because of relabeling function
    tensor_specs_str: str    # ab|cd,cd|ef->ab|ef
    tensors: Tuple[torch.Tensor]  # [ones((2,3,2,2)), ones((2,), ones((2, 2, 2, 4))]
    tensor_labels: Tuple[str]  # ['A', 'B', 'C']

    out_idx: List[chr]  # [a, b]
    contracted_idx: List[chr]  # [c, d]
    diag_idx: List[chr]  # ie, i|i kind of tensor
    in_idx: List[chr]  # [e, f]

    idx_to_out_tensors: Dict[chr, Tuple[int]]  # tensors for which this index is contravariant
    idx_to_in_tensors: Dict[chr, Tuple[int]]   # tensors for which this index is covariant
    idx_to_diag_tensors: Dict[chr, Tuple[int]] # tensors for which this index is both (co/contra)variant

    idx_to_dim: Dict[chr, int]  # {a:2, b:3, c:2, d:2, e:2, f:4}

    _einsum_str: str  # 'abcd,c,cdef->abef'

    @staticmethod
    def __legacy_init__(index_spec_list, tensors, tag=None):
        return TensorContraction(list((i, t) for (i, t) in zip(index_spec_list, tensors)))

    def __init__(self, specs: List[Tuple], label=None):
        """

        Args:
            specs: list of tensor specs, tuples like ('ab|cd', A, 'A'). 'ab' are output/contravariant indices
            label: optional label of the Tensor, used in printing

        Main job of this function is to figure out which indices get contracted and which ones are left over.
        For the left-over ones, determine whether they are input, output or both.

        For instance a_i b_i^i -> results in _i being an input index
        """
        self._original_specs = tuple(specs)

        tensor_specs: List[str] = []
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

            tensor_specs.append(tensor_spec)
            tensors.append(tensor_data)
            tensor_labels.append(tensor_label)

        self.tensor_specs = list(tensor_specs)
        self.tensors = tuple(tensors)
        self.tensor_labels = tuple(tensor_labels)
        self.label = label if label is not None else GLOBALS.generate_tensor_name()

        # see "Diagonal logic" notes
        # map tensors to dimensions
        idx_to_dim: Dict[chr, int] = {}
        idx_to_out_tensors: Dict[chr, List[int]] = {}
        idx_to_in_tensors: Dict[chr, List[int]] = {}
        idx_to_diag_tensors: Dict[chr, List[int]] = {}

        all_indices = set(''.join(tensor_specs).replace('|', ''))
        assert ' ' not in all_indices
        for idx in all_indices:
            idx_to_out_tensors.setdefault(idx, [])
            idx_to_in_tensors.setdefault(idx, [])
            idx_to_diag_tensors.setdefault(idx, [])

        # for each index, get a list of tensors for which it's contravariant-only (out), covariant-only (in) or both (diagonal)
        for tensor_id, (tensor_spec, tensor_data) in enumerate(zip(self.tensor_specs, self.tensors)):
            out_idx_term, in_idx_term = tensor_spec.split('|')
            for idx in out_idx_term:
                if idx not in in_idx_term:
                    idx_to_out_tensors[idx].append(tensor_id)
            for idx in in_idx_term:
                if idx not in out_idx_term:
                    idx_to_in_tensors[idx].append(tensor_id)
            for idx in set(out_idx_term+in_idx_term):
                if idx in out_idx_term and idx in in_idx_term:
                    idx_to_diag_tensors[idx].append(tensor_id)

            # get index shape from provided tensor
            for (idx, dim) in zip(out_idx_term + in_idx_term, tensor_data.shape):
                if idx in idx_to_dim:
                    assert idx_to_dim[idx] == dim, f"trying to set idx {idx} in {tensor_spec} to {dim}, " \
                                                   f"but it's already set to have dimension {idx_to_dim[idx]}"
                assert dim > 0, f"Index {idx} in {tensor_spec} must have positive dimension, instead see {dim}"
                idx_to_dim[idx] = dim

        self.idx_to_dim = idx_to_dim   # TODO(y): use frozendict
        self.idx_to_out_tensors = u.freeze_multimap(idx_to_out_tensors)
        self.idx_to_in_tensors = u.freeze_multimap(idx_to_in_tensors)
        self.idx_to_diag_tensors = u.freeze_multimap(idx_to_diag_tensors)

        out_idx = []
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
            elif is_in and is_diag:   # contracted with diagonal on right
                in_idx.append(idx)
            elif is_diag:  # passthrough without multiplication
                in_idx.append(idx)
                out_idx.append(idx)
            elif is_out:   # passhtrough to left without contraction
                out_idx.append(idx)
            elif is_in:    # passthrough to right without contraction
                in_idx.append(idx)
            else:
                assert False, "index is neither out, in or diagonal, how did this happen?"

        self.out_idx = out_idx
        self.contracted_idx = contracted_idx
        self.in_idx = in_idx
        self.diag_idx = [x for x in out_idx if x in in_idx]  # intersect while preserving order

        print(self)
        print(self.out_idx)
        print(self.in_idx)
        print(self.diag_idx)

        self.tensor_specs_str = f"{','.join(tensor_specs)}->{''.join(list(self.out_idx))}|{''.join(list(self.in_idx))}"

        if not self.diag_idx:    # einsum can't materialize diagonal tensors, don't generate string here
            einsum_in = ','.join(self._process_for_einsum(tensor_spec) for tensor_spec in self.tensor_specs)
            einsum_out = ''.join(self.out_idx) + ''.join(self.in_idx)
            self._einsum_spec = f'{einsum_in}->{einsum_out}'
        else:
            print("diagonal tensor, no einsum for " +','.join(self.tensor_specs))
            self._einsum_spec = None   # unsupported by torch.einsum
        print('***', self._einsum_spec)


    @staticmethod
    def _process_for_einsum(spec):
        """Processes tensor spec for einsum. Drop upper/lower convention, dedup indices occurring both as upper/lower,
        which happens for diagonal tensors"""
        out_spec, in_spec = spec.split('|')
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
        return f"{dim_spec} '{self.label}', out({','.join(self.out_idx)}), in({','.join(self.in_idx)}), spec: {','.join(self.tensor_specs)}"

    def __repr__(self):
        return self.__str__()

    def _rename_index(self, old_name, new_name):
        if old_name == new_name:
            return

        print(f'renaming {old_name} to {new_name}')
        def rename_dictionary_entry(d: Dict[chr, Any], old_name: chr, new_name: chr):
            if old_name not in d:
                return
            assert isinstance(d, dict)
            assert new_name not in d
            assert isinstance(old_name, str) and len(old_name) == 1
            assert isinstance(new_name, str) and len(new_name) == 1
            d[new_name] = d[old_name]
            del d[old_name]

        def rename_list_entry(ll, old_name, new_name):  # {len(l.count(old_name)}!=1
            if old_name not in ll:
                return
            assert isinstance(ll, list)
            assert isinstance(old_name, str)
            assert len(old_name) == 1
            assert isinstance(new_name, str)
            assert len(new_name) == 1

            assert ll.count(old_name) == 1, f"Found  {ll.count(old_name)} instances of {old_name} in {ll}"
            pos = ll.index(old_name)
            ll[pos] = new_name

        assert new_name not in self.all_indices, f"Renaming '{old_name}' to '{new_name}' but '{new_name}' already used in " \
                                                 f"tensor {str(self)}"

        rename_list_entry(self.out_idx, old_name, new_name)
        rename_list_entry(self.in_idx, old_name, new_name)
        rename_list_entry(self.contracted_idx, old_name, new_name)
        # tensor_specs: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
        for i, index_spec in enumerate(self.tensor_specs):
            self.tensor_specs[i] = index_spec.replace(old_name, new_name)  # ab|c -> de|c
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
        return GLOBALS.all_indices_list[largest_pos+1:largest_pos+1+count]

    # def _generate_unused_indices(self, other: 'ContractibleTensor2', count=1):
    #     """Generate first count indices which aren't being used in current or other tensor"""
    #     assert count >= 1
    #
    #     self_indices = set(self.in_idx + self.out_idx + self.contracted_idx)
    #     other_indices = set(other.in_idx + other.out_idx + other.contracted_idx)
    #     used_indices = self_indices.union(other_indices)
    #     unused_indices = GLOBALS.all_indices.difference(used_indices)
    #     return tuple(sorted(unused_indices))[:count]

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
        assert len(left.in_idx) >= len(right.out_idx), "only allow incomplete contraction on left tensor, right tensor must contract all " \
                                                       "output indices"

        # rename indices of right to avoid clashes
        if len(right.contracted_idx + right.in_idx):
            print('before step 1 rename')
            print(right.tensor_specs)

            max_renames = len(right.contracted_idx + right.in_idx)
            new_indices = left._generate_unused_indices(other=right, count=max_renames)
            rename_count = 0
            left_uncontracted_in_idx = left.in_idx[len(right.out_idx):]
            taken_indices = set(left.in_idx + left.contracted_idx + left.out_idx)  # this indices are used by LEFT tensor
            for idx in right.contracted_idx + right.in_idx:
                # rename all indices of RIGHT that conflict with LEFT unless they are in right's out index set (happens for diagonal)
                if idx in taken_indices:
                    right._rename_index(idx, new_indices[rename_count])
                    rename_count += 1

        # match right's out indices to left's in indices
        # left tensor may have more indices than right, this happens in Hessian-vector product
        print('before step 2 rename', left.tensor_specs, right.tensor_specs)
        left_contracted = left.in_idx[:len(right.out_idx)]  # contract these in-indices of LEFT with all out-indices of RIGHT
        print(f'matching left {left_contracted} to right {right.out_idx}')
        for left_idx, right_idx in zip(left_contracted, right.out_idx):
            right._rename_index(right_idx, left_idx)
        print('after step 2 rename')
        print(right.tensor_specs)

        # TODO: here (add ..)
        new_specs = self._transpose_specs(left.tensor_specs + right.tensor_specs, left.tensors + right.tensors,
                                          left.tensor_labels + right.tensor_labels)
        result = TensorContraction(new_specs, label=f"{self.label}*{other.label}")
        print(f'contracting {self.label} and {other.label}')
        print(','.join(self.tensor_specs) + ' * ' + ','.join(other.tensor_specs) + ' = ' + ','.join(result.tensor_specs))
        return result

    @staticmethod
    def _transpose_specs(tensor_specs, tensors, tensor_labels):
        return list((spec, tensor, label) for (spec, tensor, label) in zip(tensor_specs, tensors, tensor_labels))

    @staticmethod
    def from_dense_vector(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object corresponding to given dense vector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        idx0 = GLOBALS.idx0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord(idx0), ord(idx0) + len(x.shape)))
        return TensorContraction.__legacy_init__([idx + '|'], [x], tag)

    @staticmethod
    def from_dense_covector(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object corresponding to given dense covector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        idx0 = GLOBALS.idx0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord(idx0), ord(idx0) + len(x.shape)))
        return TensorContraction.__legacy_init__(['|' + idx], [x], tag)

    # TODO(y): label/tag parameters fix
    @staticmethod
    def from_dense_matrix(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object treating it as linear map (1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        idx0 = GLOBALS.idx0
        idx1 = chr(ord(idx0)+1)
        return TensorContraction.__legacy_init__([idx0 + '|' + idx1], [x], tag)

    @staticmethod
    def from_dense_tensor(tensor_spec: str, x: torch.Tensor, label: str = None):
        return TensorContraction([(tensor_spec, x, label)])

    @staticmethod
    def from_diag_matrix(x: torch.Tensor, tag: str = None):
        """Creates ContractibleTensor corresponding to diagonal matrix"""
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        assert x.shape[0] > 0
        idx0 = GLOBALS.idx0
        return TensorContraction.__legacy_init__([idx0 + '|' + idx0], [x], tag)

    @property
    def value(self):
        if not self._einsum_spec:
            # manually materialize for diagonal matrix case, since einsum doesn't support i->ii syntax
            assert len(self.in_idx) == 1, "Only support diagonal rank-2 diagonal tensors"
            assert len(self.tensors) == 1, f"Only support diagonal rank-2 single diagonal matrix, got {len(self.tensors)}"
            return torch.diag(self.tensors[0])

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
        if new_einsum_spec != self._einsum_spec:
            print("Warning, diagonal hack")
            return torch.einsum(new_einsum_spec, *self.tensors)
        return torch.einsum(self._einsum_spec, *self.tensors)


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

class FunctionSharedImpl(ABC):
    # addition
    def __add__(self, other: 'Function'):
        assert isinstance(other, Function)
        if isinstance(self, FunctionAddition):
            return FunctionAddition(self.children + [other])
        else:
            return FunctionAddition([self, other])

    # contraction
    def __mul__(self, other: 'Function'):
        assert isinstance(other, Function)
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


class Function(ABC):
    """Differentiable function"""

    @abstractmethod
    def __call__(self, t: 'Tensor'):
        pass

    @abstractmethod
    def in_dims(self):
        """Input (lower) dimensions"""
        pass

    @abstractmethod
    def out_dims(self):
        """Output (upper) dimensions"""
        pass


class CompositeFunction(Function, FunctionSharedImpl, ABC):
    """Function defined as a combination of AtomicFunction objects using +, *, @"""
    # TODO(y): use tuple instead of list

    children: List[Function]

    # TODO(y) drop dimensions? These are are only needed at tensor level
    def out_dims(self):
        pass

    def in_dims(self):
        pass


class FunctionAddition(CompositeFunction):
    def __init__(self, children: List['Function']):
        # Must have two children. Otherwise, it's harder to tell if two functions are the same
        # ie, FunctionAddition([f]) == FunctionContraction([f])
        assert len(children) >= 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[0](t)
        for c in self.children[1:]:
            result = result + c(t)
        return result


class FunctionContraction(CompositeFunction):
    def __init__(self, children: List['Function']):
        assert len(children) >= 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[0]
        for c in self.children[1:]:
            result = result * c(t)
        return result


class FunctionComposition(CompositeFunction):
    def __init__(self, children: List['Function']):
        assert len(children) >= 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[-1]
        for c in self.children[:-1]:
            result = c(result)
        return result


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
        elif isinstance(other, Operator):  # TODO(y): maybe reuse Operator logic here
            return OperatorComposition([self, other])
        else:
            return NotImplemented

    def __call__(self, other: Function) -> Function:
        # atomic function, defer to existing derivative implementation
        if isinstance(other, AtomicFunction):
            return other.d(self.order)

        # addition rule
        elif isinstance(other, FunctionAddition):
            return FunctionAddition([self(c) for c in other.children])

        # product rule
        elif isinstance(other, FunctionContraction):
            add_children: List[Function] = []
            for (i, c1) in enumerate(other.children):
                mul_children: List[Function] = []
                dc1 = D(c1)
                for (j, c2) in enumerate(other.children):
                    if i == j:
                        mul_children.append(dc1)
                    else:
                        mul_children.append(c2)
                add_children.append(FunctionContraction(mul_children))
            return FunctionAddition(add_children)

        # chain rule
        elif isinstance(other, FunctionComposition):
            mul_children = []
            for (i, c1) in enumerate(other.children):
                mul_children.append(FunctionComposition([D(c1)] + other.children[i + 1:]))
            return FunctionContraction(mul_children)


D = D_(order=1)
D2 = D @ D
