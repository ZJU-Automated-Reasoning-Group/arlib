if __name__ == "__main__":
    import arlib.itp as kd
    import arlib.itp.config as config
    config.timing = True
    import time
    start_time = time.perf_counter()
    modules = []
    def mark(tag):
        global start_time
        elapsed_time = time.perf_counter() - start_time
        modules.append((elapsed_time, tag))
        start_time = time.perf_counter()
    start_time = time.perf_counter()
    import arlib.itp.all
    mark("arlib.itp.all")
    import arlib.itp.theories.real as R
    mark("real")
    import arlib.itp.theories.bitvec as bitvec
    mark("bitvec")
    import arlib.itp.theories.real.complex as complex
    mark("complex")
    import arlib.itp.theories.algebra.group as group
    mark("group")
    import arlib.itp.theories.algebra.lattice
    mark("lattice")
    import arlib.itp.theories.algebra.ordering
    mark("ordering")
    import arlib.itp.theories.bool as bool_
    mark("bool")
    import arlib.itp.theories.int
    mark("int")
    import arlib.itp.theories.real.interval
    mark("interval")
    import arlib.itp.theories.seq as seq
    mark("seq")
    import arlib.itp.theories.set
    mark("set")
    import arlib.itp.theories.fixed
    mark("fixed")
    import arlib.itp.theories.float
    mark("float")
    import arlib.itp.theories.real.arb
    mark("arb")
    import arlib.itp.theories.real.sympy
    mark("sympy")
    import arlib.itp.theories.nat
    mark("nat")
    import arlib.itp.theories.real.vec
    mark("vec")
    import arlib.itp.theories.logic.intuitionistic
    mark("intuitionistic")
    import arlib.itp.theories.logic.temporal
    mark("temporal")

    print("\n========= Module import times ========\n")
    for (elapsed_time, tag) in sorted(modules, reverse=True):
        print(f"{elapsed_time:.6f} {tag}")

    import itertools
    for tag, group in itertools.groupby(sorted(config.perf_log, key=lambda x: x[0]), key=lambda x: x[0]):
        print("\n=============" + tag + "=============\n")
        for (tag, data, time) in sorted(group, key=lambda x: x[2], reverse=True)[:20]:
            print(f"{time:.6f}: {data}")
