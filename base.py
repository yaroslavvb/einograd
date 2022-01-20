"""Base types used everywhere"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

import more_itertools
import natsort
import opt_einsum as oe
import torch
from attrdict import AttrDict
from opt_einsum import helpers as oe_helpers

gl = AttrDict({'DEBUG': True, 'device': 'cpu', 'PURE_TENSOR_NETWORKS': False,
               'tensor_count': 0, 'ALLOW_PARTIAL_CONTRACTIONS': False,
               'ALLOW_UNSORTED_INDICES': False})


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

    out_indices: List[chr]
    in_indices: List[chr]
    contracted_indices: List[chr]

    index_spec_list: List[str]  # ['ij|k', 'k|lm'] corresponding to out|in
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
    def out_dims(self):
        return tuple(self.index_dim[c] for c in self.out_indices)

    @property
    def in_dims(self):
        return tuple(self.index_dim[c] for c in self.in_indices)

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
            if not gl.ALLOW_PARTIAL_CONTRACTIONS:
                assert self.in_indices[-1] == chr(ord(self.in_indices[0]) + len(self.in_indices) - 1)

    def __str__(self):
        out_dims0 = tuple(self.index_dim[c] for c in self.out_indices)
        assert out_dims0 == self.out_dims

        in_dims0 = tuple(self.index_dim[c] for c in self.in_indices)
        assert in_dims0 == self.in_dims

        out_dim_spec = ','.join(str(d) for d in out_dims0)
        in_dim_spec = ','.join(str(d) for d in in_dims0)
        dim_spec = f"{out_dim_spec}|{in_dim_spec}"
        return f"{dim_spec} {self.tag}, out: {','.join(self.out_indices)}, in: {','.join(self.in_indices)}, spec: {','.join(self.index_spec_list)}"

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

        assert new_name not in (self.out_indices + self.contracted_indices + self.in_indices)

        rename_list_entry(self.out_indices, old_name, new_name)
        rename_list_entry(self.in_indices, old_name, new_name)
        rename_list_entry(self.contracted_indices, old_name, new_name)
        # _index_spec_list: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
        for i, index_spec in enumerate(self.index_spec_list):
            self.index_spec_list[i] = index_spec.replace(old_name, new_name)
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

    def __init__(self, index_spec_list, tensors, tag=None):
        """['ij|k', 'k|lm'], [tensor1, tensor2]"""

        if tag is not None:
            self.tag = tag
        else:
            self.tag = f'tensor{gl.tensor_count:02d}'
            gl.tensor_count += 1

        index_spec_list = index_spec_list.copy()

        if len(index_spec_list) != len(tensors):
            print(f"Provided {len(tensors)} tensors, but your index spec has {len(index_spec_list)} terms: ")
            for (i, term) in enumerate(index_spec_list):
                print(f"term {i:2d}: {term:>20}")
                assert False

        self.index_spec_list = index_spec_list
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
            all_indices_term = output_indices + input_indices

            # special handling for diagonal tensors
            if output_indices == input_indices:
                self.IS_DIAGONAL = True
            else:
                self.IS_DIAGONAL = False
                assert len(all_indices_term) == len(set(all_indices_term))
            if gl.PURE_TENSOR_NETWORKS:  # this disallows diagonal tensors
                assert not set(input_indices).intersection(set(output_indices))

            all_indices.update(set(all_indices_term))

            for idx in output_indices:
                # noinspection PyTypeChecker
                self.index_out_tensors.setdefault(idx, []).append(tensor)
                self.index_out_degree[idx] = self.index_out_degree.get(idx, 0) + 1
            for idx in input_indices:
                # noinspection PyTypeChecker
                self.index_in_tensors.setdefault(idx, []).append(tensor)
                self.index_in_degree[idx] = self.index_in_degree.get(idx, 0) + 1

            for idx in set(all_indices_term):
                self.index_degree[idx] = self.index_degree.get(idx, 0) + 1

            for (idx, dim) in zip(all_indices_term, tensor.shape):
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
                if gl.PURE_TENSOR_NETWORKS:
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
        # this should fail for diagonal
        assert not set(self.out_indices).intersection(self.in_indices)

        self._out_dims = tuple(self.index_dim[c] for c in self.out_indices)
        self._in_dims = tuple(self.index_dim[c] for c in self.in_indices)

        einsum_in = ','.join(index_spec.replace('|', '') for index_spec in self.index_spec_list)
        einsum_out = ''.join(self.out_indices) + ''.join(self.in_indices)

        # special case for i->ii tensor
        if self.IS_DIAGONAL:
            assert len(tensors) == 1
            assert len(einsum_out) == 2
            self._einsum_spec = f'{einsum_out}->{einsum_out}'
        else:
            self._einsum_spec = f'{einsum_in}->{einsum_out}'


    def __mul__(self, other: 'ContractibleTensor'):
        """Contraction operation"""
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

        left = ContractibleTensor(self.index_spec_list, self.tensors)
        right = ContractibleTensor(other.index_spec_list, other.tensors)

        # assert len(set(left.in_indices).union(right.out_indices)) > 0, "Outer products not supported"

        # first increment (never decrement because of _check_indices_sorted invariant) indices on the right to match
        # inputs
        assert len(right.out_indices) <= len(left.in_indices)

        # special handling for outer products
        if len(left.in_indices) == 0:
            incr1 = len(set(left.out_indices + left.contracted_indices))
        else:
            incr1 = ord(left.in_indices[0]) - ord(right.out_indices[0])
        if not gl.ALLOW_UNSORTED_INDICES:
            assert incr1 >= 0, f"Problem matching right tensor's {right.out_indices} to left tensor's {left.in_indices}, " \
                           f"we are assuming right tensors indices are incremented, never decremented"

        for idx in reversed(sorted(set(right.in_indices + right.out_indices + right.contracted_indices))):
            if incr1 > 0:
                right.rename_index(idx, chr(ord(idx)+incr1))

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
                left.rename_index(idx, chr(ord(idx)+offset))

        # print('my new spec list', left._index_spec_list)
        # print('right new spec list', right._index_spec_list)

        result = StructuredTensor(left._index_spec_list + right._index_spec_list, left.tensors + right.tensors)
        print(f'contracting {self.tag} and {other.tag}')
        print(','.join(self._index_spec_list) + ' * ' + ','.join(other._index_spec_list) + ' = ' + ','.join(result._index_spec_list))
        return result



    @property
    def value(self):
        if self.IS_DIAGONAL:  # torch.einsum doesn't support 'i->ii' kind of einsum, do it manually
            assert len(self.in_indices) == 1, "Only support diagonal rank-2 tensors"
            assert len(self.tensors) == 1
            return torch.diag(self.tensors[0])

        # hack to deal with diagonal tensors
        # ein_in, ein_out = self._einsum_spec.split('->')
        #        new_terms = []
        # for term in ein_in.split(','):
        #     if len(term) == 2 and term[0] == term[1]:
        #        new_term = term[0]
        #    else:
        #        new_term = term
        #    new_terms.append(new_term)

        # new_einsum_spec = ','.join(new_terms) + '->' + ein_out
        # if new_einsum_spec != self._einsum_spec:
        #     print("Warning, diagonal hack")
        #    return torch.einsum(new_einsum_spec, *self.tensors)
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
