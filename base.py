"""Base types used everywhere"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

import more_itertools
import natsort
import opt_einsum as oe
import torch
from opt_einsum import helpers as oe_helpers


#GLOBALS = AttrDict({'DEBUG': True, 'device': 'cpu', 'PURE_TENSOR_NETWORKS': False,
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
        self.ALLOW_PARTIAL_CONTRACTIONS = True   # allow some indices of the left tensor to remain uncontracted
        self.ALLOW_UNSORTED_INDICES = False
        self.MAX_INDEX_COUNT = 1000
        self.idx0 = 'a'
        self.all_indices = set(chr(ord(self.idx0) + i) for i in range(self.MAX_INDEX_COUNT))

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


class ContractibleTensor(Tensor):
    tag: str  # tag helpful for debugging
    _in_dims: Tuple[int]
    _out_dims: Tuple[int]

    out_idx: List[chr]
    in_idx: List[chr]
    contracted_idx: List[chr]
    out_idx_to_tensors: Dict[chr, List[torch.Tensor]]  # d['i'] == [tensor1, tensor2, ...]
    in_idx_to_tensors: Dict[chr, List[torch.Tensor]]

    idx_to_dim: Dict[chr, int]
    "index dimensions, ie index_dim['i']==3"

    tensor_specs: List[str]  # ['ij|k', 'k|lm'] corresponding to out|in
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
        assert more_itertools.is_sorted(self.in_idx, strict=True)
        if not self.IS_DIAGONAL and not GLOBALS.ALLOW_PARTIAL_CONTRACTIONS:
            assert more_itertools.is_sorted(self.out_idx + self.contracted_idx + self.in_idx, strict=True)
        elif not GLOBALS.ALLOW_PARTIAL_CONTRACTIONS:
            # new "in" indices may come from partially unconctracted indices
            # of left tensor, losing alphabetical ordering
            assert more_itertools.is_sorted(self.out_idx + self.contracted_idx + self.in_idx, strict=False)

        # check that output/input indices are consecutive
        if self.out_idx:
            assert self.out_idx[-1] == chr(ord(self.out_idx[0]) + len(self.out_idx) - 1)
        # input indices won't be consecutive after partial contractions
        if self.in_idx:
            if not GLOBALS.ALLOW_PARTIAL_CONTRACTIONS:
                assert self.in_idx[-1] == chr(ord(self.in_idx[0]) + len(self.in_idx) - 1)

    def __str__(self):
        out_dims0 = tuple(self.idx_to_dim[c] for c in self.out_idx)
        assert out_dims0 == self.out_dims

        in_dims0 = tuple(self.idx_to_dim[c] for c in self.in_idx)
        assert in_dims0 == self.in_dims

        out_dim_spec = ','.join(str(d) for d in out_dims0)
        in_dim_spec = ','.join(str(d) for d in in_dims0)
        dim_spec = f"{out_dim_spec}|{in_dim_spec}"
        return f"{dim_spec} '{self.tag}', out({','.join(self.out_idx)}), in({','.join(self.in_idx)}), spec: {','.join(self.tensor_specs)}"

    def rename_index(self, old_name, new_name):
        # print(f"naming {tag}:{old_name} to {new_name}")

        if old_name == new_name:
            return

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

        def rename_list_entry(ll, old_name, new_name):  # {len(l.count(old_name)}!=1
            if old_name not in ll:
                return
            assert isinstance(ll, list)
            assert isinstance(old_name, str)
            assert len(old_name) == 1
            assert isinstance(new_name, str)
            assert len(new_name) == 1

            # assert l.count(old_name) == 1, f"Found  {l.count(old_name)} instances of {old_name} in {l}"
            pos = ll.index(old_name)
            ll[pos] = new_name

        assert new_name not in self.all_indices, f"Renaming '{old_name}' to '{new_name}' but '{new_name}' already used in " \
                                                 f"tensor {str(self)}"

        rename_list_entry(self.out_idx, old_name, new_name)
        rename_list_entry(self.in_idx, old_name, new_name)
        rename_list_entry(self.contracted_idx, old_name, new_name)
        # _index_spec_list: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
        for i, index_spec in enumerate(self.tensor_specs):
            self.tensor_specs[i] = index_spec.replace(old_name, new_name)
        #  _einsum_spec: str  # 'ij,jk->ik'
        self._einsum_spec = self._einsum_spec.replace(old_name, new_name)
        rename_dictionary_entry(self.index_degree, old_name, new_name)
        rename_dictionary_entry(self.index_out_degree, old_name, new_name)
        rename_dictionary_entry(self.index_in_degree, old_name, new_name)
        rename_dictionary_entry(self.out_idx_to_tensors, old_name, new_name)
        rename_dictionary_entry(self.in_idx_to_tensors, old_name, new_name)
        rename_dictionary_entry(self.idx_to_dim, old_name, new_name)
        rename_list_entry(self.out_idx, old_name, new_name)
        rename_list_entry(self.in_idx, old_name, new_name)
        rename_list_entry(self.contracted_idx, old_name, new_name)

    @property
    def all_indices(self):
        return tuple(self.out_idx + self.contracted_idx + self.in_idx)

    def __init__(self, index_spec_list: List[str], tensors: List[torch.Tensor], tag=None):
        """['ij|k', 'k|lm'], [tensor1, tensor2]"""

        assert isinstance(index_spec_list, List),  f'must provide a list of index specs, but got type ' \
                                                   f'{type(index_spec_list)} '
        assert isinstance(tensors, List), f'must provide a list of tensors, but got type {type(tensors)}'

        if tag is not None:
            self.tag = tag
        else:
            self.tag = f'tensor{GLOBALS.tensor_count:02d}'
            GLOBALS.tensor_count += 1

        index_spec_list = index_spec_list.copy()

        # special handling of diagonal matrices
        self.IS_DIAGONAL = False
        if len(index_spec_list) == 1:
            _temp_in,_temp_out = index_spec_list[0].split('|')
            if len(_temp_in) == 1 and _temp_in == _temp_out:
                self.IS_DIAGONAL = True

        if len(index_spec_list) != len(tensors):
            print(f"Provided {len(tensors)} tensors, but your index spec has {len(index_spec_list)} terms: ")
            for (i, term) in enumerate(index_spec_list):
                print(f"term {i:2d}: {term:>20}")
                assert False

        self.tensor_specs = index_spec_list
        self.tensors = tensors
        self.index_degree = {}
        self.index_out_degree = {}
        self.index_in_degree = {}
        self.out_idx_to_tensors = {}
        self.in_idx_to_tensors = {}

        all_indices = set()  # all

        # create dict of sizes, by matching indices to tensors
        index_dim = {}  # ['ij'], [np.ones((2,5))] gives {'i': 2, 'j': 5}
        HAS_DIAGONAL = False
        for (index_spec, tensor) in zip(index_spec_list, tensors):
            assert isinstance(index_spec, str)
            assert isinstance(tensor, torch.Tensor), f"Provided not an instance of torch.Tensor, {index_spec}, {tensor}"
            output_indices, input_indices = index_spec.split('|')
            all_indices_term = output_indices + input_indices

            # special handling for diagonal tensors
            if output_indices == input_indices:
                HAS_DIAGONAL = True
            else:
                assert len(all_indices_term) == len(set(all_indices_term))
            if GLOBALS.PURE_TENSOR_NETWORKS:  # this disallows diagonal tensors
                assert not set(input_indices).intersection(set(output_indices))

            all_indices.update(set(all_indices_term))

            for idx in output_indices:
                # noinspection PyTypeChecker
                self.out_idx_to_tensors.setdefault(idx, []).append(tensor)
                self.index_out_degree[idx] = self.index_out_degree.get(idx, 0) + 1
            for idx in input_indices:
                # noinspection PyTypeChecker
                self.in_idx_to_tensors.setdefault(idx, []).append(tensor)
                self.index_in_degree[idx] = self.index_in_degree.get(idx, 0) + 1

            for idx in set(all_indices_term):
                self.index_degree[idx] = self.index_degree.get(idx, 0) + 1

            for (idx, dim) in zip(all_indices_term, tensor.shape):
                if idx in index_dim:
                    assert index_dim[idx] == dim, f"trying to set idx {idx} in indices {index_spec} to {dim}, " \
                                                  f"but it's already set to have dimension {index_dim[idx]}"
                assert dim > 0, f"Index {idx} has dimension {dim}"
                index_dim[idx] = dim
        self.idx_to_dim = index_dim

        # sanity check, for each index make sure it appears equal number of times as contravariant and covariant
        self.contracted_idx = []
        self.out_idx = []
        self.in_idx = []
        for idx in sorted(all_indices):
            # number of tensors for which this idx is upper/contravariant
            out_count = len(self.out_idx_to_tensors.get(idx, []))
            # number of tensors for which this idx is lower/covariant
            in_count = len(self.in_idx_to_tensors.get(idx, []))
            assert out_count == self.index_out_degree.get(idx, 0)
            assert in_count == self.index_in_degree.get(idx, 0)

            if self.IS_DIAGONAL:
                self.in_idx.append(idx)
                self.out_idx.append(idx)
                assert self.index_out_degree[idx] == 1
                assert self.index_in_degree[idx] == 1

            elif out_count and in_count:
                if not HAS_DIAGONAL:
                    assert out_count == in_count
                if GLOBALS.PURE_TENSOR_NETWORKS:
                    assert out_count == 1  # in pure tensor networks, each index is contracted at most once
                else:
                    assert out_count <= 2, f"Index {idx} is contravariant in {out_count} tensors, suspicious," \
                                           f"it should be 1 for regular tensors, and 2 for diagonal matrices "
                assert idx not in self.contracted_idx, f"Trying to add {idx} as contracted index twice"
                self.contracted_idx.append(idx)
                self._check_indices_sorted()

            elif out_count and not in_count:
                assert idx not in self.out_idx, f"Trying to add {idx} as output index twice"
                self.out_idx.append(idx)
            elif in_count and not out_count:
                assert idx not in self.out_idx, f"Trying to add {idx} as input index twice"
                self.in_idx.append(idx)
            else:
                assert False, f"Shouldn't be here, {idx} is marked as occuring {out_count} times as contravariant " \
                              f"and {in_count} as covariant"

        assert len(self.out_idx) == len(set(self.out_idx))
        assert len(self.in_idx) == len(set(self.in_idx))
        # this should fail for diagonal
        if not self.IS_DIAGONAL:
            assert not set(self.out_idx).intersection(self.in_idx)

        self._out_dims = tuple(self.idx_to_dim[c] for c in self.out_idx)
        self._in_dims = tuple(self.idx_to_dim[c] for c in self.in_idx)

        einsum_in = ','.join(index_spec.replace('|', '') for index_spec in self.tensor_specs)
        einsum_out = ''.join(self.out_idx) + ''.join(self.in_idx)
        # import pdb; pdb.set_trace()


        #if self.IS_DIAGONAL:
        #    self._einsum_spec = f'{input_indices}->{einsum_in}'
        #    self.in_indices = list(input_indices)
        #    self.out_indices = list(output_indices)

        # special case for i->ii tensor
        # get 'ii->ii'
        if self.IS_DIAGONAL:
            assert len(tensors) == 1
            assert len(einsum_out) == 2
            in_indices = self.in_idx
            assert len(in_indices) == 1
            self._einsum_spec = f'{in_indices[0]}->{einsum_out}'
            #            import pdb; pdb.set_trace()
        else:
            self._einsum_spec = f'{einsum_in}->{einsum_out}'
        print('***', self._einsum_spec)

    def generate_unused_indices(self, other: 'ContractibleTensor', count=1):
        """Generate first count indices which aren't being used in current or other tensor"""
        assert count>=1
        self_indices = set(self.in_idx + self.out_idx + self.contracted_idx)
        other_indices = set(other.in_idx + other.out_idx + other.contracted_idx)
        used_indices = self_indices.union(other_indices)
        unused_indices = GLOBALS.all_indices.difference(used_indices)
        return tuple(sorted(unused_indices))[:count]

    def __mul__(self, other: 'ContractibleTensor'):
        print(f'contracting {self} with {other}')
        """Contraction operation"""
        self._check_indices_sorted()
        other._check_indices_sorted()

        left = ContractibleTensor(self.tensor_specs, self.tensors, tag=f'cloned_{self.tag}')
        right = ContractibleTensor(other.tensor_specs, other.tensors, tag=f'cloned_{other.tag}')

        assert len(left.in_idx) >= len(right.out_idx)

        # rename indices of right to avoid clashes
        if len(right.contracted_idx + right.in_idx):
            print('before rename')
            print(str(right))
            new_indices = self.generate_unused_indices(other=other, count=len(right.contracted_idx + right.in_idx))
            rename_count = 0
            taken_indices = set(left.in_idx + left.contracted_idx + left.out_idx)
            for idx in right.contracted_idx + right.in_idx:
                if idx in taken_indices:
                    right.rename_index(idx, new_indices[rename_count])
                    rename_count += 1
            print('after rename')
            print(str(right))

        # match right's out indices to left's in indices
        # left tensor may have more indices than right, to support Hessian contractions
        indices_to_contract = right.out_idx[:len(left.in_idx)]
        for left_idx, right_idx in zip(left.in_idx, indices_to_contract):
            print(left_idx, '-', right_idx)
            right.rename_index(right_idx, left_idx)

        result = ContractibleTensor(left.tensor_specs + right.tensor_specs, left.tensors + right.tensors,
                                    tag=f"{self.tag}*{other.tag}")
        print(f'contracting {self.tag} and {other.tag}')
        print(','.join(self.tensor_specs) + ' * ' + ','.join(other.tensor_specs) + ' = ' + ','.join(result.tensor_specs))
        return result

    @staticmethod
    def from_dense_vector(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object corresponding to given dense vector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord('i'), ord('i')+len(x.shape)))
        return ContractibleTensor([idx + '|'], [x], tag)

    @staticmethod
    def from_dense_covector(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object corresponding to given dense covector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord('i'), ord('i')+len(x.shape)))
        return ContractibleTensor(['|' + idx], [x], tag)

    @staticmethod
    def from_dense_matrix(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object (LinearMap with 1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        return ContractibleTensor(['i|j'], [x], tag)

    @property
    def value(self):
        if self.IS_DIAGONAL:  # torch.einsum doesn't support 'i->ii' kind of einsum, do it manually
            assert len(self.in_idx) == 1, "Only support diagonal rank-2 tensors"
            assert len(self.tensors) == 1
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

    @staticmethod
    def from_diag_matrix(x: torch.Tensor, tag: str = None):
        """Creates ContractibleTensor corresponding to diagonal matrix"""
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        assert x.shape[0] > 0
        return ContractibleTensor(['i|i'], [x], tag)


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
