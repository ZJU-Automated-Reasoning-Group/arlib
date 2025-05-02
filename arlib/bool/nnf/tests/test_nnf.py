import copy
import pathlib
import pickle
import platform
import shutil
import threading
import types
import typing as t

import pytest

from hypothesis import (assume, event, given, strategies as st, settings,
                        HealthCheck)

import arlib.bool.nnf as nnf

from arlib.bool.nnf import (Var, And, Or, amc, dimacs, dsharp, operators,
                 tseitin, complete_models, config, pysat, all_models)

memoized = [
    method
    for method in vars(nnf.NNF).values()
    if isinstance(method, types.FunctionType) and hasattr(method, "memo")
]
assert memoized, "No memoized methods found, did the implementation change?"
for method in memoized:
    method.set = lambda *args: None  # type: ignore

settings.register_profile('patient', deadline=2000,
                          suppress_health_check=(HealthCheck.too_slow,))
settings.load_profile('patient')

a, b, c = Var('a'), Var('b'), Var('c')

fig1a = (~a & b) | (a & ~b)
fig1b = (~a | ~b) & (a | b)

satlib = pathlib.Path(__file__).parent / "testdata" / "satlib"
uf20 = [dsharp.load(file.open()) for file in (satlib / "uf20").glob("*.nnf")]
uf20_cnf = [
    dimacs.load(file.open()) for file in (satlib / "uf20").glob("*.cnf")
]  # type: t.List[And[Or[Var]]]

# Test config default value before the tests start mucking with the state
assert config.sat_backend == "auto"


def test_all_models_basic():
    assert list(nnf.all_models([])) == [{}]
    assert list(nnf.all_models([1])) == [{1: False}, {1: True}]
    assert len(list(nnf.all_models(range(10)))) == 1024


@given(st.sets(st.integers(), max_size=8))
def test_all_models(names):
    result = list(nnf.all_models(names))
    # Proper result size
    assert len(result) == 2**len(names)
    # Only real names, only booleans
    assert all(name in names and isinstance(value, bool)
               for model in result
               for name, value in model.items())
    # Only complete models
    assert all(len(model) == len(names)
               for model in result)
    # No duplicate models
    assert len({tuple(model.items()) for model in result}) == len(result)


def test_basic():
    assert a.satisfied_by(dict(a=True))
    assert (a | b).satisfied_by(dict(a=False, b=True))
    assert not (a & b).satisfied_by(dict(a=True, b=False))

    assert (a & b).satisfiable()
    assert not (a & ~a).satisfiable()
    assert not (a & (~a & b)).satisfiable()

    assert ((a | b) & (b | c)).satisfiable()


def test_amc():
    assert amc.NUM_SAT(fig1a) == 2
    assert amc.NUM_SAT(fig1b) == 4

    assert amc.GRAD(a, {'a': 0.5}, 'a') == (0.5, 1)


names = st.integers(1, 8)


@st.composite
def variables(draw):
    return Var(draw(names), draw(st.booleans()))


@st.composite
def booleans(draw):
    return draw(st.sampled_from((nnf.true, nnf.false)))


@st.composite
def leaves(draw):
    return draw(st.one_of(variables(), booleans()))


@st.composite
def terms(draw):
    return And(Var(name, draw(st.booleans()))
               for name in draw(st.sets(names)))


@st.composite
def clauses(draw):
    return Or(Var(name, draw(st.booleans()))
              for name in draw(st.sets(names)))


@st.composite
def DNF(draw):
    return Or(draw(st.frozensets(terms())))


@st.composite
def CNF(draw):
    sentence = And(draw(st.frozensets(clauses())))
    return sentence


@st.composite
def models(draw):
    return And(
        Var(name, draw(st.booleans()))
        for name in range(1, draw(st.integers(min_value=1, max_value=9)))
    )


@st.composite
def MODS(draw):
    num = draw(st.integers(min_value=1, max_value=9))
    amount = draw(st.integers(min_value=0, max_value=10))
    return Or(And(Var(name, draw(st.booleans()))
                  for name in range(1, num))
              for _ in range(amount))


@st.composite
def internal(draw, children):
    return draw(st.sampled_from((And, Or)))(draw(st.frozensets(children)))


@st.composite
def NNF(draw):
    return draw(st.recursive(variables(), internal))


@st.composite
def DNNF(draw):
    sentence = draw(NNF())
    assume(sentence.decomposable())
    return sentence


@given(DNF())
def test_hyp(sentence: nnf.Or):
    assume(len(sentence.children) != 0)
    assume(sentence.decomposable())
    assert sentence.satisfiable()
    assert sentence.vars() <= set(range(1, 9))


@given(MODS())
def test_MODS(sentence: nnf.Or):
    assert sentence.smooth()
    assert sentence.flat()
    assert sentence.decomposable()
    assert sentence.simply_conjunct()


@given(MODS())
def test_MODS_satisfiable(sentence: nnf.Or):
    if len(sentence.children) > 0:
        assert sentence.satisfiable()
    else:
        assert not sentence.satisfiable()


@pytest.fixture(scope='module', params=[True, False])
def merge_nodes(request):
    return request.param


@given(sentence=DNNF())
def test_DNNF_sat_strategies(sentence: nnf.NNF, merge_nodes):
    sat = sentence.satisfiable()
    if sat:
        assert sentence.simplify(merge_nodes) != nnf.false
        assert amc.SAT(sentence)
        event("Sentence satisfiable")
    else:
        assert sentence.simplify(merge_nodes) == nnf.false
        assert not amc.SAT(sentence)
        event("Sentence not satisfiable")


def test_amc_numsat():
    for sentence in uf20:
        assert (amc.NUM_SAT(sentence.make_smooth())
                == len(list(sentence.models())))


@given(sentence=NNF())
def test_idempotent_simplification(sentence: nnf.NNF, merge_nodes):
    sentence = sentence.simplify(merge_nodes)
    assert sentence.simplify(merge_nodes) == sentence


@given(sentence=NNF())
def test_simplify_preserves_meaning(sentence: nnf.NNF, merge_nodes):
    simple = sentence.simplify(merge_nodes)
    assert sentence.equivalent(simple)
    for model in sentence.models():
        assert simple.satisfied_by(model)
    for model in simple.models():
        assert sentence.condition(model).simplify(merge_nodes) == nnf.true


@given(sentence=NNF())
def test_simplify_eliminates_bools(sentence: nnf.NNF, merge_nodes):
    assume(sentence != nnf.true and sentence != nnf.false)
    if any(node == nnf.true or node == nnf.false
           for node in sentence.walk()):
        event("Sentence contained booleans originally")
    sentence = sentence.simplify(merge_nodes)
    if sentence == nnf.true or sentence == nnf.false:
        event("Sentence simplified to boolean")
    else:
        for node in sentence.walk():
            assert node != nnf.true and node != nnf.false


@given(NNF())
def test_simplify_merges_internal_nodes(sentence: nnf.NNF):
    if any(any(type(node) == type(child)
               for child in node.children)
           for node in sentence.walk()
           if isinstance(node, nnf.Internal)):
        event("Sentence contained immediately mergeable nodes")
        # Nodes may also be merged after intermediate nodes are removed
    for node in sentence.simplify().walk():
        if isinstance(node, nnf.Internal):
            for child in node.children:
                assert type(node) != type(child)


@given(sentence=DNNF())
def test_simplify_solves_DNNF_satisfiability(sentence: nnf.NNF, merge_nodes):
    if sentence.satisfiable():
        event("Sentence is satisfiable")
        assert sentence.simplify(merge_nodes) != nnf.false
    else:
        event("Sentence is not satisfiable")
        assert sentence.simplify(merge_nodes) == nnf.false


def test_dimacs_sat_serialize():
    # http://www.domagoj-babic.com/uploads/ResearchProjects/Spear/dimacs-cnf.pdf
    sample_input = """c Sample SAT format
c
p sat 4
(*(+(1 3 -4)
   +(4)
   +(2 3)))
"""
    assert dimacs.loads(sample_input) == And({
        Or({Var(1), Var(3), ~Var(4)}),
        Or({Var(4)}),
        Or({Var(2), Var(3)})
    })


@pytest.mark.parametrize(
    'serialized, sentence',
    [
        ('p sat 2\n(+((1)+((2))))', Or({Var(1), Or({Var(2)})}))
    ]
)
def test_dimacs_sat_weird_input(serialized: str, sentence: nnf.NNF):
    assert dimacs.loads(serialized) == sentence


def test_dimacs_cnf_serialize():
    sample_input = """c Example CNF format file
c
p cnf 4 3
1 3 -4 0
4 0 2
-3
"""
    assert dimacs.loads(sample_input) == And({
        Or({Var(1), Var(3), ~Var(4)}),
        Or({Var(4)}),
        Or({Var(2), ~Var(3)})
    })


def test_dimacs_rejects_weird_digits():
    with pytest.raises(dimacs.DecodeError):
        dimacs.loads("p cnf 1 1\n¹ 0")


@given(NNF())
def test_arbitrary_dimacs_sat_serialize(sentence: nnf.NNF):
    assert dimacs.loads(dimacs.dumps(sentence)) == sentence
    # Removing spaces may change the meaning, but shouldn't make it invalid
    # At least as far as our parser is concerned, a more sophisticated one
    # could detect variables with too high names
    serial = dimacs.dumps(sentence).split('\n')
    serial[1] = serial[1].replace(' ', '')
    dimacs.loads('\n'.join(serial))


@given(CNF())
def test_arbitrary_dimacs_cnf_serialize(sentence: And[Or[Var]]):
    reloaded = dimacs.loads(dimacs.dumps(sentence, mode='cnf'))
    assert reloaded.is_CNF()
    assert reloaded == sentence


@given(NNF())
def test_dimacs_cnf_serialize_accepts_only_cnf(sentence: nnf.NNF):
    if sentence.is_CNF():
        event("CNF sentence")
        dimacs.dumps(sentence, mode='cnf')
    else:
        event("Not CNF sentence")
        with pytest.raises(dimacs.EncodeError):
            dimacs.dumps(sentence, mode='cnf')


@pytest.mark.parametrize(
    'fname, clauses',
    [
        ('bf0432-007.cnf', 3667),
        ('sw100-1.cnf', 3100),
        ('uuf250-01.cnf', 1065),
        ('uf20-01.cnf', 90),
    ]
)
def test_cnf_benchmark_data(fname: str, clauses: int):
    with (satlib / fname).open() as f:
        sentence = dimacs.load(f)
    assert isinstance(sentence, And) and len(sentence.children) == clauses


def test_dsharp_output():
    with (satlib / "uf20-01.nnf").open() as f:
        sentence = dsharp.load(f)
    with (satlib / "uf20-01.cnf").open() as f:
        clauses = dimacs.load(f)
    assert sentence.decomposable()
    # this is not a complete check, but clauses.models() is very expensive
    assert all(clauses.satisfied_by(model) for model in sentence.models())


@given(NNF())
def test_walk_unique_nodes(sentence: nnf.NNF):
    result = list(sentence.walk())
    assert len(result) == len(set(result))
    assert len(result) <= sentence.size() + 1


@given(st.dictionaries(st.integers(), st.booleans()))
def test_to_model(model: dict):
    sentence = nnf.And(nnf.Var(k, v) for k, v in model.items())
    assert sentence.to_model() == model


@given(DNNF())
def test_models_deterministic_sanity(sentence: nnf.NNF):
    """Running _models_deterministic() on a non-deterministic decomposable
    sentence may return duplicate models but should not return unsatisfying
    models and should return each satisfying model at least once.
    """
    assert model_set(sentence._models_decomposable()) == model_set(
        sentence._models_deterministic()
    )


def test_models_deterministic_trivial():
    assert list(nnf.true._models_deterministic()) == [{}]
    assert list(nnf.false._models_deterministic()) == []
    assert list(a._models_deterministic()) == [{"a": True}]


@pytest.mark.parametrize(
    'sentence, size',
    [
        ((a & b), 2),
        (a & (a | b), 4),
        ((a | b) & (~a | ~b), 6),
        (And({
            Or({a, b}),
            And({a, Or({a, b})}),
        }), 6)
    ]
)
def test_size(sentence: nnf.NNF, size: int):
    assert sentence.size() == size


@pytest.mark.parametrize(
    'a, b, contradictory',
    [
        (a, ~a, True),
        (a, b, False),
        (a, a, False),
        (a & b, a & ~b, True),
        (a & (a | b), b, False),
        (a & (a | b), ~a, True),
    ]
)
def test_contradicts(a: nnf.NNF, b: nnf.NNF, contradictory: bool):
    assert a.contradicts(b) == contradictory


@given(NNF())
def test_false_contradicts_everything(sentence: nnf.NNF):
    assert nnf.false.contradicts(sentence)


@given(DNNF())
def test_equivalent(sentence: nnf.NNF):
    assert sentence.equivalent(sentence)
    assert sentence.equivalent(sentence | nnf.false)
    assert sentence.equivalent(sentence & (nnf.Var('A') | ~nnf.Var('A')))
    if sentence.satisfiable():
        assert not sentence.equivalent(sentence & nnf.false)
        assert not sentence.equivalent(sentence & nnf.Var('A'))
    else:
        assert sentence.equivalent(sentence & nnf.false)
        assert sentence.equivalent(sentence & nnf.Var('A'))


@given(NNF(), NNF())
def test_random_equivalent(a: nnf.NNF, b: nnf.NNF):
    if a.vars() != b.vars():
        if a.equivalent(b):
            event("Equivalent, different vars")
            assert b.equivalent(a)
            for model in a.models():
                assert b.condition(model).valid()
            for model in b.models():
                assert a.condition(model).valid()
        else:
            event("Not equivalent, different vars")
            assert (any(not b.condition(model).valid()
                        for model in a.models()) or
                    any(not a.condition(model).valid()
                        for model in b.models()))
    else:
        if a.equivalent(b):
            event("Equivalent, same vars")
            assert b.equivalent(a)
            assert model_set(a.models()) == model_set(b.models())
        else:
            event("Not equivalent, same vars")
            assert model_set(a.models()) != model_set(b.models())


@given(NNF())
def test_smoothing(sentence: nnf.NNF):
    if not sentence.smooth():
        event("Sentence not smooth yet")
        smoothed = sentence.make_smooth()
        assert type(sentence) is type(smoothed)
        assert smoothed.smooth()
        assert sentence.equivalent(smoothed)
        assert smoothed.make_smooth() == smoothed
    else:
        event("Sentence already smooth")
        assert sentence.make_smooth() == sentence


def hashable_dict(model):
    return frozenset(model.items())


def model_set(model_gen):
    return frozenset(map(hashable_dict, model_gen))


def test_uf20_models():

    for sentence in uf20:
        assert sentence.decomposable()
        m = list(sentence._models_decomposable())
        models = model_set(m)
        assert len(m) == len(models)
        assert models == model_set(sentence._models_deterministic())


def test_instantiating_base_classes_fails():
    with pytest.raises(TypeError):
        nnf.NNF()
    with pytest.raises(TypeError):
        nnf.Internal()
    with pytest.raises(TypeError):
        nnf.Internal({nnf.Var(3)})


@given(NNF())
def test_negation(sentence: nnf.NNF):
    n_vars = len(sentence.vars())
    models_orig = model_set(sentence.models())
    models_negated = model_set(sentence.negate().models())
    assert len(models_orig) + len(models_negated) == 2**n_vars
    assert len(models_orig | models_negated) == 2**n_vars


@given(NNF())
def test_model_counting(sentence: nnf.NNF):
    assert sentence.model_count() == len(list(sentence.models()))


def test_uf20_model_counting():
    for sentence in uf20:
        nnf.NNF._deterministic_sentences.pop(id(sentence), None)
        assert sentence.model_count() == len(list(sentence.models()))
        sentence.mark_deterministic()
        assert sentence.model_count() == len(list(sentence.models()))


@given(NNF())
def test_validity(sentence: nnf.NNF):
    if sentence.valid():
        event("Valid sentence")
        assert all(sentence.satisfied_by(model)
                   for model in nnf.all_models(sentence.vars()))
    else:
        event("Invalid sentence")
        assert any(not sentence.satisfied_by(model)
                   for model in nnf.all_models(sentence.vars()))


def test_uf20_validity():
    for sentence in uf20:
        nnf.NNF._deterministic_sentences.pop(id(sentence), None)
        assert not sentence.valid()
        sentence.mark_deterministic()
        assert not sentence.valid()


@given(CNF())
def test_is_CNF(sentence: nnf.NNF):
    assert sentence.is_CNF()
    assert sentence.is_CNF(strict=True)
    assert not sentence.is_DNF()


def test_is_CNF_examples():
    assert And().is_CNF()
    assert And().is_CNF(strict=True)
    assert And({Or()}).is_CNF()
    assert And({Or()}).is_CNF(strict=True)
    assert And({Or({a, ~b})}).is_CNF()
    assert And({Or({a, ~b})}).is_CNF(strict=True)
    assert And({Or({a, ~b}), Or({c, ~c})}).is_CNF()
    assert not And({Or({a, ~b}), Or({c, ~c})}).is_CNF(strict=True)


@given(DNF())
def test_is_DNF(sentence: nnf.NNF):
    assert sentence.is_DNF()
    assert sentence.is_DNF(strict=True)
    assert not sentence.is_CNF()


def test_is_DNF_examples():
    assert Or().is_DNF()
    assert Or().is_DNF(strict=True)
    assert Or({And()}).is_DNF()
    assert Or({And()}).is_DNF(strict=True)
    assert Or({And({a, ~b})}).is_DNF()
    assert Or({And({a, ~b})}).is_DNF(strict=True)
    assert Or({And({a, ~b}), And({c, ~c})}).is_DNF()
    assert not Or({And({a, ~b}), And({c, ~c})}).is_DNF(strict=True)


@given(NNF())
def test_to_MODS(sentence: nnf.NNF):
    assume(len(sentence.vars()) <= 5)
    mods = sentence.to_MODS()
    assert mods.is_MODS()
    assert mods.is_DNF()
    assert mods.is_DNF(strict=True)
    assert mods.smooth()
    assert isinstance(mods, Or)
    assert mods.model_count() == len(mods.children)


@given(MODS())
def test_is_MODS(sentence: nnf.NNF):
    assert sentence.is_MODS()


@given(NNF())
def test_pairwise(sentence: nnf.NNF):
    new = sentence.make_pairwise()
    assert new.equivalent(sentence)
    if new not in {nnf.true, nnf.false}:
        assert all(len(node.children) == 2
                   for node in new.walk()
                   if isinstance(node, nnf.Internal))


@given(NNF())
def test_implicates(sentence: nnf.NNF):
    implicates = sentence.implicates()
    assert implicates.equivalent(sentence)
    assert implicates.is_CNF(strict=True)
    assert not any(a.children < b.children
                   for a in implicates.children
                   for b in implicates.children)


@given(NNF())
def test_implicants(sentence: nnf.NNF):
    implicants = sentence.implicants()
    assert implicants.equivalent(sentence)
    assert implicants.is_DNF()
    assert not any(a.children < b.children
                   for a in implicants.children
                   for b in implicants.children)


@given(NNF())
def test_implicants_idempotent(sentence: nnf.NNF):
    assume(len(sentence.vars()) <= 6)
    implicants = sentence.implicants()
    implicates = sentence.implicates()
    assert implicants.implicants() == implicants
    assert implicates.implicants() == implicants


@given(NNF())
def test_implicates_implicants_negation_rule(sentence: nnf.NNF):
    """Any implicate is also a negated implicant of the negated sentence.

    .implicates() gives some implicates, and .implicants() gives all
    implicants.

    So sentence.negate().implicants().negate() gives all implicates,
    and sentence.negate().implicates().negate() gives some implicants.
    """
    assume(sentence.size() <= 30)
    assert (
        sentence.negate().implicants().negate().children
        >= sentence.implicates().children
    )
    assert (
        sentence.negate().implicates().negate().children
        <= sentence.implicants().children
    )


def test_implicates_implicants_negation_rule_example():
    """These failed an old version of the previous test. See issue #3."""
    sentence = Or({And({~Var(1), Var(2)}), And({~Var(3), Var(1)})})
    assert (
        sentence.negate().implicants().negate().children
        >= sentence.implicates().children
    )
    assert (
        sentence.negate().implicates().negate().children
        <= sentence.implicants().children
    )


@given(NNF(), NNF())
def test_implies(a: nnf.NNF, b: nnf.NNF):
    if a.implies(b):
        event("Implication")
        for model in a.models():
            assert b.condition(model).valid()
    else:
        event("No implication")
        assert any(not b.condition(model).valid()
                   for model in a.models())


def test_uf20_cnf_sat():
    for sentence in uf20_cnf:
        assert sentence.is_CNF()
        assert sentence.satisfiable()
        # It would be nice to compare .models() output to another algorithm
        # But even 20 variables is too much
        # So let's just hope that test_cnf_sat does enough
        at_least_one = False
        for model in sentence.models():
            assert sentence.satisfied_by(model)
            at_least_one = True
        assert at_least_one


@given(NNF(), NNF())
def test_xor(a: nnf.NNF, b: nnf.NNF):
    c = operators.xor(a, b)
    for model in nnf.all_models(c.vars()):
        assert (a.satisfied_by(model) ^ b.satisfied_by(model) ==
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_nand(a: nnf.NNF, b: nnf.NNF):
    c = operators.nand(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) and b.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_nor(a: nnf.NNF, b: nnf.NNF):
    c = operators.nor(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) or b.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_implies2(a: nnf.NNF, b: nnf.NNF):
    c = operators.implies(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) and not b.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_implied_by(a: nnf.NNF, b: nnf.NNF):
    c = operators.implied_by(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((b.satisfied_by(model) and not a.satisfied_by(model)) !=
                c.satisfied_by(model))


@given(NNF(), NNF())
def test_iff(a: nnf.NNF, b: nnf.NNF):
    c = operators.iff(a, b)
    for model in nnf.all_models(c.vars()):
        assert ((a.satisfied_by(model) == b.satisfied_by(model)) ==
                c.satisfied_by(model))


@given(NNF())
def test_forget(sentence: nnf.NNF):
    # Assumption to reduce the time in testing
    assume(sentence.size() <= 15)

    # Test that forgetting a backbone variable doesn't change the theory
    T = sentence & Var('added_var')
    assert sentence.equivalent(T.forget({'added_var'}))

    # Test the tseitin projection
    assert sentence.equivalent(sentence.to_CNF().forget_aux())

    # Test that models of a projected theory are consistent with the original
    names = list(sentence.vars())[:2]
    T = sentence.forget(names)
    assert not any([v in T.vars() for v in names])

    for m in T.models():
        assert sentence.condition(m).satisfiable()


@given(NNF())
def test_project(sentence: nnf.NNF):
    # Test that we get the same as projecting and forgetting
    assume(len(sentence.vars()) > 3)
    vars1 = list(sentence.vars())[:2]
    vars2 = list(sentence.vars())[2:]
    assert sentence.forget(vars1).equivalent(sentence.project(vars2))


@given(NNF())
def test_pickling(sentence: nnf.NNF):
    new = pickle.loads(pickle.dumps(sentence))
    assert sentence == new
    assert sentence is not new
    assert sentence.object_count() == new.object_count()


@given(NNF())
def test_copying_does_not_copy(sentence: nnf.NNF):
    assert sentence is copy.copy(sentence) is copy.deepcopy(sentence)
    assert copy.deepcopy([sentence])[0] is sentence


if shutil.which('dsharp') is not None:
    def test_dsharp_compile_uf20():
        sentence = uf20_cnf[0]
        compiled = dsharp.compile(sentence)
        compiled_smooth = dsharp.compile(sentence, smooth=True)
        assert sentence.equivalent(compiled)
        assert sentence.equivalent(compiled_smooth)
        assert compiled.decomposable()
        assert compiled_smooth.decomposable()
        assert compiled_smooth.smooth()

    @given(CNF())
    def test_dsharp_compile(sentence: And[Or[Var]]):
        compiled = dsharp.compile(sentence)
        compiled_smooth = dsharp.compile(sentence, smooth=True)
        assert compiled.decomposable()
        assert compiled_smooth.decomposable()
        assert compiled_smooth.smooth()
        if sentence.satisfiable():  # See nnf.dsharp.__doc__
            assert sentence.equivalent(compiled)
            assert sentence.equivalent(compiled_smooth)

    @given(CNF())
    def test_dsharp_compile_converting_names(sentence: And[Or[Var]]):
        sentence = And(Or(Var(str(var.name), var.true) for var in clause)
                       for clause in sentence)
        compiled = dsharp.compile(sentence)
        assert all(isinstance(name, str) for name in compiled.vars())
        if sentence.satisfiable():
            assert sentence.equivalent(compiled)


def test_mark_deterministic():
    s = And()
    t = And()

    assert not s.marked_deterministic()
    assert not t.marked_deterministic()

    s.mark_deterministic()

    assert s.marked_deterministic()
    assert not t.marked_deterministic()

    t.mark_deterministic()

    assert s.marked_deterministic()
    assert t.marked_deterministic()

    del s

    assert t.marked_deterministic()


@given(NNF())
def test_tseitin(sentence: nnf.NNF):

    # Assumption to reduce the time in testing
    assume(sentence.size() <= 10)

    T = tseitin.to_CNF(sentence)
    assert T.is_CNF()
    assert T.is_CNF(strict=True)
    assert tseitin.to_CNF(T) == T
    assert T.forget_aux().equivalent(sentence)

    models = list(complete_models(T.models(), sentence.vars() | T.vars()))

    for mt in models:
        assert sentence.satisfied_by(mt)

    assert len(models) == sentence.model_count()


@given(CNF())
def test_tseitin_preserves_CNF(sentence: And[Or[Var]]):
    assert sentence.to_CNF() == sentence


def test_tseitin_required_detection():
    assert a.to_CNF() == And({Or({a})})
    assert And().to_CNF() == And()
    assert Or().to_CNF() == And({Or()})
    assert (a | b).to_CNF() == And({a | b})
    assert And({a | b, b | c}).to_CNF() == And({a | b, b | c})
    assert And({And({Or({And({~a})})})}).to_CNF() == And({Or({~a})})


@given(models())
def test_complete_models(model: nnf.And[nnf.Var]):
    m = {v.name: v.true for v in model}
    neg = {v.name: not v.true for v in model}

    zero = list(complete_models([m], model.vars()))
    assert len(zero) == 1

    one = list(complete_models([m], model.vars() | {"test1"}))
    assert len(one) == 2

    two = list(complete_models([m], model.vars() | {"test1", "test2"}))
    assert len(two) == 4
    assert all(x.keys() == m.keys() | {"test1", "test2"} for x in two)

    if m:
        multi = list(
            complete_models([m, neg], model.vars() | {"test1", "test2"})
        )
        assert len(multi) == 8
        assert len({frozenset(x.items()) for x in multi}) == 8  # all unique
        assert all(x.keys() == m.keys() | {"test1", "test2"} for x in multi)


if (platform.uname().system, platform.uname().machine) == ('Linux', 'x86_64'):

    @config(sat_backend="kissat")
    def test_kissat_uf20():
        for sentence in uf20_cnf:
            assert sentence.satisfiable()

    @config(sat_backend="kissat")
    @given(CNF())
    def test_kissat_cnf(sentence: And[Or[Var]]):
        assert sentence.satisfiable() == sentence._cnf_satisfiable_native()

    @config(sat_backend="kissat")
    @given(NNF())
    def test_kissat_nnf(sentence: And[Or[Var]]):
        assert (
            sentence.satisfiable()
            == tseitin.to_CNF(sentence)._cnf_satisfiable_native()
        )


@config(sat_backend="auto")
def test_config():
    assert config.sat_backend == "auto"

    # Imperative style works
    config.sat_backend = "native"
    assert config.sat_backend == "native"

    # Context manager works
    with config(sat_backend="kissat"):
        assert config.sat_backend == "kissat"
    assert config.sat_backend == "native"

    # Bad values are caught
    with pytest.raises(ValueError):
        config.sat_backend = "invalid"

    # In context managers too, before we enter
    with pytest.raises(ValueError):
        with config(sat_backend="invalid"):
            assert False

    config.sat_backend = "kissat"
    assert config.sat_backend == "kissat"

    # Old value is restored when we leave, even if changed inside
    # (this may or may not be desirable behavior, but if it changes
    # we should know)
    with config(sat_backend="native"):
        assert config.sat_backend == "native"
        config.sat_backend = "auto"
        assert config.sat_backend == "auto"
    assert config.sat_backend == "kissat"

    # Bad settings are caught
    with pytest.raises(AttributeError):
        config.invalid = "somevalue"

    # Decorator works
    @config(sat_backend="native")
    def somefunc(recurse=False):
        assert config.sat_backend == "native"
        if recurse:
            # Even if we call it again while it's in progress
            config.sat_backend = "auto"
            somefunc(recurse=False)
            assert config.sat_backend == "auto"

    somefunc()
    assert config.sat_backend == "kissat"
    somefunc(recurse=True)
    assert config.sat_backend == "kissat"

    # Context managers can be reused and nested without getting confused
    reentrant_cm = config(sat_backend="auto")
    assert config.sat_backend == "kissat"
    with reentrant_cm:
        assert config.sat_backend == "auto"
        config.sat_backend = "native"
        with reentrant_cm:
            assert config.sat_backend == "auto"
        assert config.sat_backend == "native"
    assert config.sat_backend == "kissat"


@config(sat_backend="auto")
def test_config_multithreading():
    # Settings from one thread don't affect another
    config.sat_backend = "native"

    def f():
        assert config.sat_backend == "auto"
        config.sat_backend = "kissat"
        assert config.sat_backend == "kissat"

    thread = threading.Thread(target=f)
    thread.start()
    thread.join()

    assert config.sat_backend == "native"


@given(NNF())
def test_solve(sentence: nnf.NNF):
    solution = sentence.solve()
    if solution is None:
        assert not sentence.satisfiable()
    else:
        assert sentence.satisfiable()
        assert sentence.satisfied_by(solution)


@pytest.fixture(
    scope="module",
    params=[
        "cadical",
        "glucose30",
        "glucose41",
        "lingeling",
        "maplechrono",
        "maplecm",
        "maplesat",
        "minicard",
        "minisat22",
        "minisat-gh",
    ],
)
def pysat_solver(request):
    return config(pysat_solver=request.param)


if pysat.available:

    @given(sentence=CNF())
    def test_pysat_satisfiable(sentence: And[Or[Var]], pysat_solver):
        with pysat_solver:
            assert sentence._cnf_satisfiable_native() == pysat.satisfiable(
                sentence
            )

    @given(sentence=CNF())
    def test_pysat_models(sentence: And[Or[Var]], pysat_solver):
        native_models = list(sentence._cnf_models_native())
        with pysat_solver:
            pysat_models = list(pysat.models(sentence))
        native_set = model_set(native_models)
        pysat_set = model_set(pysat_models)
        assert native_set == pysat_set
        assert (
            len(native_models)
            == len(pysat_models)
            == len(native_set)
            == len(pysat_set)
        )

    @given(sentence=CNF())
    def test_pysat_solve(sentence: And[Or[Var]], pysat_solver):
        with pysat_solver:
            native_solution = sentence._cnf_solve()
            pysat_solution = pysat.solve(sentence)
            if native_solution is None:
                assert pysat_solution is None
                assert not sentence._cnf_satisfiable_native()
                assert not pysat.satisfiable(sentence)
            else:
                assert pysat_solution is not None
                assert sentence.satisfied_by(native_solution)
                assert sentence.satisfied_by(pysat_solution)
                assert sentence._cnf_satisfiable_native()
                assert pysat.satisfiable(sentence)

    def test_pysat_uf20(pysat_solver):
        with pysat_solver:
            for sentence in uf20_cnf:
                assert pysat.satisfiable(sentence)
                solution = pysat.solve(sentence)
                assert solution
                assert sentence.satisfied_by(solution)


@given(NNF())
def test_satisfiable(sentence: nnf.NNF):
    assert sentence.satisfiable() == any(
        sentence.satisfied_by(model) for model in all_models(sentence.vars())
    )


@given(NNF())
def test_models(sentence: nnf.NNF):
    real_models = [
        model
        for model in all_models(sentence.vars())
        if sentence.satisfied_by(model)
    ]
    models = list(sentence.models())
    assert len(real_models) == len(models)
    assert model_set(real_models) == model_set(models)


@given(NNF())
def test_toCNF_simplification_names(sentence: nnf.NNF):
    names1 = set(sentence.vars())
    T = sentence.to_CNF(simplify=False)
    names2 = set({v for v in T.vars() if not isinstance(v, nnf.Aux)})
    assert names1 == names2


def test_toCNF_simplification():
    x = Var("x")
    T = x | ~x

    T1 = T.to_CNF()
    T2 = T.to_CNF(simplify=False)

    assert T1 == nnf.true
    assert T1.is_CNF()

    assert T2 != nnf.true
    assert T2 == And({Or({~x, x})})
    assert T2.is_CNF()


@config(auto_simplify=False)
def test_nesting():
    a, b, c, d, e, f = Var("a"), Var("b"), Var("c"), Var("d"), \
                       Var("e"), Var("f")

    # test left nestings on And
    config.auto_simplify = False
    formula = a & (b & c)
    formula = formula & (d | e)
    assert formula == And({And({And({b, c}), a}), Or({d, e})})
    config.auto_simplify = True
    formula = a & (b & c)
    formula = formula & (d | e)
    assert formula == And({a, b, c, Or({d, e})})

    # test right nestings on And
    config.auto_simplify = False
    formula = a & (b & c)
    formula = (d | e) & formula
    assert formula == And({And({And({b, c}), a}), Or({d, e})})
    config.auto_simplify = True
    formula = a & (b & c)
    formula = (d | e) & formula
    assert formula == And({a, b, c, Or({d, e})})

    # test nestings on both sides with And
    config.auto_simplify = False
    formula = a & (b & c)
    formula2 = d & (e & f)
    formula = formula & formula2
    assert formula == And({(And({a, And({b, c})})), And({d, And({e, f})})})
    config.auto_simplify = True
    formula = a & (b & c)
    formula2 = d & (e & f)
    formula = formula & formula2
    assert formula == And({a, b, c, d, e, f})

    # test left nestings on Or
    config.auto_simplify = False
    formula = a | (b | c)
    formula = formula | (d & e)
    assert formula == Or({Or({Or({b, c}), a}), And({d, e})})
    config.auto_simplify = True
    formula = a | (b | c)
    formula = formula | (d & e)
    assert formula == Or({a, b, c, And({d, e})})

    # test right nestings on Or
    config.auto_simplify = False
    formula = a | (b | c)
    formula = (d & e) | formula
    assert formula == Or({Or({Or({b, c}), a}), And({d, e})})
    config.auto_simplify = True
    formula = a | (b | c)
    formula = (d & e) | formula
    assert formula == Or({a, b, c, And({d, e})})

    # test nestings on both sides with Or
    config.auto_simplify = False
    formula = a | (b | c)
    formula2 = d | (e | f)
    formula = formula | formula2
    assert formula == Or({(Or({a, Or({b, c})})), Or({d, Or({e, f})})})
    config.auto_simplify = True
    formula = a | (b | c)
    formula2 = d | (e | f)
    formula = formula | formula2
    assert formula == Or({a, b, c, d, e, f})

    # test nestings with both And and Or
    config.auto_simplify = False
    formula = a & (b | c)
    formula2 = d & (e & f)
    formula = formula | formula2
    assert formula == Or({(And({a, Or({b, c})})), And({d, And({e, f})})})
    config.auto_simplify = True
    formula = a & (b | c)
    formula2 = d & (e & f)
    formula = formula | formula2
    assert formula == Or({(And({a, Or({b, c})})), And({d, e, f})})
