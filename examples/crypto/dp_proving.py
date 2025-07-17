"""
Proving differential privacy properties of DP algorithms using SMT solving
"""

from z3 import *
import math, sys, time

def dp_model(eps, delta=0):
    x1, x2, y1, y2 = Reals('x1 x2 y1 y2')
    adj = Abs(x1 - x2) <= 1
    if delta == 0:
        dp = Implies(adj, Abs(y1 - x1) - Abs(y2 - x2) <= eps)
    else:
        dp = Implies(adj, Or(Abs(y1 - x1) - Abs(y2 - x2) <= eps, And(y1 == y2, delta > 0)))
    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}, [dp]

def verify_dp(mech, eps, delta=0):
    vars, dp_cons = dp_model(eps, delta)
    s = Solver(); s.add(dp_cons)
    if mech == "gaussian":
        sigma = Real('sigma'); vars['sigma'] = sigma
        s.add(sigma < math.sqrt(2 * math.log(1.25/delta)) / eps)
    elif mech == "randomized_response":
        p = Real('p'); vars['p'] = p
        s.add(p == eps / (1 + eps))
    s.push(); s.add(Not(And(dp_cons)))
    t0 = time.time(); res = s.check(); t = time.time() - t0
    return (False, s.model(), vars, t) if res == sat else (True, None, vars, t)

def print_result(is_dp, model, vars, mech, eps, delta, t):
    dp_type = f"({eps},{delta})-DP" if delta > 0 else f"{eps}-DP"
    print(f"{mech.capitalize()} {dp_type}: {t:.4f}s", end=' ')
    if is_dp:
        print("✓ Satisfies DP")
    else:
        print("✗ Violation found. Counterexample:")
        for k, v in vars.items():
            if model and v in model: print(f"  {k} = {model[v]}")

def main():
    mechs = {"laplace": "Laplace", "gaussian": "Gaussian", "randomized_response": "Randomized Response"}
    mech = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in mechs else "laplace"
    print(f"Verifying: {mechs[mech]}")
    if mech == "laplace":
        for eps in [0.1, 0.5, 1.0]:
            print_result(*verify_dp(mech, eps), mech, eps, 0, verify_dp(mech, eps)[3])
    elif mech == "gaussian":
        for eps, delta in [(1.0, 0.01), (0.5, 0.05), (0.1, 0.1)]:
            print_result(*verify_dp(mech, eps, delta), mech, eps, delta, verify_dp(mech, eps, delta)[3])
    elif mech == "randomized_response":
        for eps in map(math.log, [3, 9, 19]):
            print_result(*verify_dp(mech, eps), mech, eps, 0, verify_dp(mech, eps)[3])

if __name__ == "__main__":
    main()
