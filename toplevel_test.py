import sys

import pytest
import torch

import util
from layers import *


def test_test():
    print('passed')

class ObsoleteContractibleTensor(Tensor):
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

        assert isinstance(index_spec_list, List), f'must provide a list of index specs, but got type ' \
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
            _temp_in, _temp_out = index_spec_list[0].split('|')
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

        # if self.IS_DIAGONAL:
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

    def generate_unused_indices(self, other: 'ObsoleteContractibleTensor', count=1):
        """Generate first count indices which aren't being used in current or other tensor"""
        assert count >= 1
        self_indices = set(self.in_idx + self.out_idx + self.contracted_idx)
        other_indices = set(other.in_idx + other.out_idx + other.contracted_idx)
        used_indices = self_indices.union(other_indices)
        unused_indices = GLOBALS.all_indices.difference(used_indices)
        return tuple(sorted(unused_indices))[:count]

    def __mul__(self, other: 'ObsoleteContractibleTensor'):
        print(f'contracting {self} with {other}')
        """Contraction operation"""
        self._check_indices_sorted()
        other._check_indices_sorted()

        left = ObsoleteContractibleTensor(self.tensor_specs, self.tensors, tag=f'cloned_{self.tag}')
        right = ObsoleteContractibleTensor(other.tensor_specs, other.tensors, tag=f'cloned_{other.tag}')

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

        result = ObsoleteContractibleTensor(left.tensor_specs + right.tensor_specs, left.tensors + right.tensors,
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
            idx = ''.join(chr(i) for i in range(ord('i'), ord('i') + len(x.shape)))
        return ObsoleteContractibleTensor([idx + '|'], [x], tag)

    @staticmethod
    def from_dense_covector(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object corresponding to given dense covector"""
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] > 0
        if idx is None:
            idx = ''.join(chr(i) for i in range(ord('i'), ord('i') + len(x.shape)))
        return ObsoleteContractibleTensor(['|' + idx], [x], tag)

    @staticmethod
    def from_dense_matrix(x: torch.Tensor, tag: str = None, idx: str = None):
        """Creates StructuredTensor object (LinearMap with 1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        return ObsoleteContractibleTensor(['i|j'], [x], tag)

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
        return ObsoleteContractibleTensor(['i|i'], [x], tag)


class LinearFunction(Function, Tensor, ABC):
    """Linear function with mixed Tensor, both upper and lower indices"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


def test_dense():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    # U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    x = DenseVector(x0)
    u.check_equal(W(x).value, W0 @ x0)

    dW = D(W)  # derivative of linear layer
    print(dW(zero) * x)  # get
    u.check_equal(dW(zero) * x, W0 @ x0)


def test_contract():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    W = DenseLinear(W0)
    U = DenseLinear(U0)
    x0 = u.to_pytorch([1, 2])
    x = DenseVector(x0)
    y = DenseCovector(x0)

    u.check_equal(W * U, W0 @ U0)
    assert isinstance(W * U, LinearMap)
    u.check_equal(W * x, [-3, 5])
    assert isinstance(W * x, Vector)
    u.check_equal(y * W, [-5, 6])
    assert isinstance(y * W, Covector)
    u.check_equal(y * x, 5)
    assert isinstance(y * x, Scalar)


def _create_unit_test_a():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Relu(x0.shape[0])
    loss = LeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss


def _create_unit_test_a_sigmoid():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Sigmoid(x0.shape[0])
    loss = LeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss


def test_unit_test_a():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == LinearLayer
    assert type(h4) == LeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    u.check_equal(a1, [1, 2])
    u.check_equal(a2, [-3, 5])
    u.check_equal(a3, [0, 5])
    u.check_equal(a4, [-30, 40])
    u.check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    u.check_equal(dh1(a1), W0)
    u.check_equal(dh2(a2), [[0, 0], [0, 1]])
    u.check_equal(dh3(a3), [[5, -6], [-7, 8]])
    u.check_equal(dh4(a4), [-30, 40])
    u.check_equal(dh4(a4) * dh3(a3), [-430, 500])
    u.check_equal(dh4(a4) * dh3(a3) * dh2(a2), [0, 500])
    u.check_equal(dh4(a4) * dh3(a3) * dh2(a2) * dh1(a1), [-1500, 2000])

    u.reset_global_forward_flops()
    assert u.get_global_forward_flops() == 0

    result = f(x)
    u.check_equal(result, 1250)
    assert u.get_global_forward_flops() == 4
    _unused_result = f(x)
    assert u.get_global_forward_flops() == 4

    # creating new composition does not reuse cache
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    _unused_result = f(x)
    assert u.get_global_forward_flops() == 2 * 4

    # partial composition test
    u.reset_global_forward_flops()
    print('flops ', u.get_global_forward_flops())
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    # result = f(x)
    a2 = f[3:](x)  # input into h2
    assert u.get_global_forward_flops() == 1
    u.check_equal(a2, [-3, 5])

    a4 = f[1:](x)  #
    assert u.get_global_forward_flops() == 3
    u.check_equal(a4, [-30, 40])

    a5 = f[:](x)  #
    assert u.get_global_forward_flops() == 4
    u.check_equal(a5, 1250)

    a5 = f[0:](x)  #
    assert u.get_global_forward_flops() == 4
    u.check_equal(a5, 1250)

    #  next steps
    # call, "D" operator,


def test_sigmoid():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    _unused_a3 = h2(a2)

    nonlin = Sigmoid(x0.shape[0])
    print('d sigmoid', D(nonlin)(a2))
    print('d2 sigmoid', D2(nonlin)(a2))
    print(D2(nonlin).order)

    u.check_close(nonlin(a2), [0.0474259, 0.993307])
    u.check_close(D(nonlin)(a2), [[0.0451767, 0], [0, 0.00664806]])
    u.check_close(D2(nonlin)(a2), [[[0.0408916, 0], [0, 0]], [[0, 0], [0, -0.00655907]]])

    assert isinstance(D2(nonlin)(a2), SymmetricBilinearMap)


def test_relu():
    f = Relu(2)
    df = f.d1  # also try D(f)
    # TODO(y): arguments to functions don't have Tensor semantics, so change type
    result = df(DenseVector([-3, 5]))
    u.check_equal(result, [[0, 0], [0, 1]])

    df = D(f)
    result = df(DenseVector([-3, 5]))
    u.check_equal(result, [[0, 0], [0, 1]])


def test_least_squares():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    _unused_a5 = h4(a4)

    assert isinstance(D(h4)(a4), Covector)
    assert isinstance(D2(h4)(a4), QuadraticForm)
    u.check_equal(D(h4)(a4), a4)
    u.check_equal(D2(h4)(a4), torch.eye(2))


def test_contraction():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == LinearLayer
    assert type(h4) == LeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)

    f(a1)  # run once to save activations

    u.check_equal(a1, [1, 2])
    u.check_equal(a2, [-3, 5])
    u.check_equal(a3, [0, 5])
    u.check_equal(a4, [-30, 40])
    u.check_equal(a5, 1250)

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
    u.check_equal(a, x00 @ y00)
    assert a.flops == 2 * d ** 3

    x = OldStructuredTensor(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    u.check_equal(x, x00 @ y00 @ z00)

    x = OldStructuredTensor(['a|b'], [x00])
    y = OldStructuredTensor(['a|b'], [y00])
    z = OldStructuredTensor(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    x.contract(y)
    xyz = x * y * z
    assert xyz.flops == 4 * d ** 3
    u.check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = OldStructuredTensor.from_dense_vector(x00, 'col')
    row = OldStructuredTensor.from_dense_covector(x00, 'row')
    mat = OldStructuredTensor.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product

    u.check_equal(row * mat * mat * mat,
                  x00 @ ma0 @ ma0 @ ma0)
    u.check_equal(mat * mat * mat * col,
                  ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    u.check_equal(mat * mat * col * row * mat * mat,
                  ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = OldStructuredTensor.from_diag_matrix(3 * x00, 'diag')
    dia0 = diag.value
    print(dia0)

    assert (row * mat * diag * mat).flops == 410  # structured reverse mode

    print()
    u.check_equal(row * mat * diag * mat,
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

    diag = TensorContraction.from_diag_matrix(3 * torch.ones((3,)), 'diag')
    assert diag.out_idx == diag.in_idx
    assert len(diag.out_idx) == 1
    dia0 = diag.value
    print(dia0)

    a = TensorContraction([('a|b', x00), ('b|c', y00)])
    u.check_equal(a.value, x00 @ y00)
    assert a.flops == 2 * d ** 3

    x = TensorContraction.__legacy_init__(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    u.check_equal(x, x00 @ y00 @ z00)

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
    u.check_equal(c, x00 @ x00)

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

    u.check_equal(c, torch.outer(rank1, rank1))

    xyz = x * y * z
    assert xyz.flops == 4 * d ** 3
    u.check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = TensorContraction.from_dense_vector(x00, 'col')
    row = TensorContraction.from_dense_covector(x00, 'row')
    mat = TensorContraction.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product

    u.check_equal(row * mat * mat * mat,
                  x00 @ ma0 @ ma0 @ ma0)
    u.check_equal(mat * mat * mat * col,
                  ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    u.check_equal(mat * mat * col * row * mat * mat,
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

    u.check_equal(row * diag, x00 @ dia0)
    u.check_equal(row * mat * diag, x00 @ ma0 @ dia0)

    result = row*mat
    print(result.ricci_str)
    assert result.ricci_str == '|a,a|b->|b'
    assert (row*mat*diag).ricci_str == '|a,a|b,b|b->|b'
    assert (row * mat * diag).ricci_str == '|a,a|b,b|b->|b'
    assert (row * mat * diag * mat).flops == 410  # structured reverse mode

    print()
    u.check_equal(row * mat * diag * mat,
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


@pytest.mark.skip(reason="this example doesn't work because our implementation currently contracts sequentially left to right only with automatic index renaming")
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

    print('====='*20)
    partial1 = A*B
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
def test_unit_test_A():

    W = Linear([[1, -2], [-3, 4]])
    U = Linear([[5, -6], [-7, 8]])
    x0 = make_vector([1, 2])
    nonlin = make_sigmoid(x0)
    loss = make_xent(x0)   # x0 used for shape inference

    (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = h4 @ h3 @ h2 @ h1  # => Composition (FunctionComposition)
    h = [None, f[3], f[2], f[1], f[0]]  # h[1] shorthand for h1, linked to a parent Composition
    assert type(h[1]) == Linear

    D(f)   # this is numerically equivalent to D(U) @ W * D(W)
    slow = D(U) @ W * D(W)
    fast = D(f) @ f[1:] * D(f[1])

    print(slow(x0).forward_flops)  # high
    print(fast(x0).forward_flops)   # low

    D(f)    # LazyDLinear # represents derivative of linear layer
    nonlin = make_sigmoid(x0)
    D(nonlin)   # LazyDRelu #

def old_test_unit_test_A():
    # W = DenseMatrix([[1, -2], [-3, 4]])
    # U = DenseMatrix([[5, -6], [-7, 8]])
    # print(W._indices())
    # x = DenseVector([1, 2])
    # loss = CrossEntropy()
    # layers = [W, nonlin, U]
    W = make_linear([[1, -2], [-3, 4]])
    U = make_linear([[5, -6], [-7, 8]])
    x0 = make_vector([1, 2])
    nonlin = make_relu(x0)
    loss = make_xent(x0)   # x0 used for shape inference

    (h1, h2, h3, h4) = (W, nonlin, U, loss)
    h = [None, W, nonlin, U, loss]  # h[1] shorthand for h1

    # TODO(y): add call count and make sure memoization is happening

    # go through Mechanics of converting this network to einsum notation
    f = h4 @ h3 @ h2 @ h1  # => Composition (FunctionComposition)
    df = D(f)  # Linear Function

    # chain rule with no memoization
    assert (D(h4) @ h3 @ h2 @ h1) * D(h3) @ h2 @ h1 * D(h2) @ h1 * D(h1) == df

    # chain rule with memoization
    D(h4)  # @ f[1:] # @ D(h3) @ f[2:] @ D(h2) @ f[3:] @ D(h1)
"""

"""
def test_present0():
    # reverse mode

    from einograd import jacobian, forward, to_expression

    W = u.create_linear([[1, -2], [-3, 4]])
    U = u.create_linear([[5, -6], [-7, 8]])
    loss_func = u.LeastSquaresLoss()
    out = to_expression(nn.Sequential(W, U, U, loss_func))
    data = u.to_pytorch([1, 2])
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


@pytest.mark.skip(reason="doesn't work yet")
def test_derivatives():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    # sum rule
    expr1 = D(W + U)
    expr2 = D(W) + D(U)
    u.check_equal(expr1(x), expr2(x))

    # product rule
    expr1 = D(W * U)
    expr2 = D(W) * U + D(U) * W
    u.check_equal(expr1(x), expr2(x))

    # chain rule
    expr1 = D(W @ U)
    expr2 = (D(W) @ U) * D(U)
    u.check_equal(expr1(x), expr2(x))

    # chain rule with memoization
    GLOBALS.function_call_count = 0
    chain = MemoizedFunctionComposition(W, U)  # TODO(y): replace with W @ U
    expr1 = D(chain)
    expr2 = (D(chain[0]) @ chain[1:]) @ D(chain[1])
    u.check_equal(expr1(x), expr2(x))
    assert GLOBALS.function_call_count == 2  # value of U(x) is requested twice, but computed once

@pytest.mark.skip(reason="doesn't work yet")
def test_present():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    hess = (D @ D)(f)
    u.check_equal(hess(x0), [[900., -1200.], [-1200., 1600.]])
    hvp = hess(x0) * x0
    u.check_equal(hvp, [-1500., 2000.])
    print(hvp.backward_flops)
    print(hvp)

    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a_sigmoid()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    hess = (D @ D)(f)
    u.check_equal(hess(x0), [[-8.62673, 13.5831], [13.5831, -22.3067]])
    hvp = hess(x0) * x0
    u.check_equal(hvp, [18.5394, -31.0303])
    print(hvp.backward_flops)
    print(hvp)


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

    assert (row*diag).ricci_str == '|a,a|a->|a'
    util.check_equal(row*diag, row0*diag0)

    assert (diag * col).ricci_str == 'a|a,a|->a|'
    u.check_equal(diag * col, diag0 * col0)

    assert (row * col).ricci_str == '|a,a|->|'
    u.check_equal(row * col, row0 @ col0)

    # weighted dot product
    assert (row * diag * col).ricci_str == '|a,a|a,a|->|'
    u.check_equal(row * diag * col, (row0 * diag0 * col0).sum())

    # Hadamard product of two diagonal matrices support combining, but not direct materialization for now, need to figure out how to deal
    # with multiple diagonal matrices, only support 1
    assert (diag * diag).ricci_str == 'a|a,a|a->a|a'
    with pytest.raises(Exception):
        print((diag*diag).value)
        u.check_equal(diag * diag, torch.diag(diag0) @ torch.diag(diag0))

    # this case could be enabled in the future, but to reduce scope currently
    # we specialize all contractions to go in left-to-right-order
    with pytest.raises(Exception):
        assert (col * diag).ricci_str == 'a|,a|a->a|'

    u.check_equal(mat * diag, ma0 @ torch.diag(diag0))
    u.check_equal(diag * mat, torch.diag(diag0) @ ma0)


def run_all():
    test_contractible_tensor2()
    test_partial_contraction_UnitTestC()
    test_contract()
    test_dense()
    test_unit_test_a()
    test_sigmoid()
    test_relu()
    test_least_squares()
    test_contraction()
    test_structured_tensor()
    test_contractible_tensor2()
    test_diagonal_problem()
    # test_derivatives()


if __name__ == '__main__':
    run_all()
    sys.exit()
    # noinspection PyTypeChecker,PyUnreachableCode
    u.run_all_tests(sys.modules[__name__])
