import random
import string
from z3 import *


class Randomness(object):
    def __init__(self, seed):
        self.seed = seed
        random.seed(self.seed)

    def get_random_integer(self, low, high):
        # high is inclusive
        return random.randint(low, high)

    def get_random_alpha_string(self, length):
        return "".join(random.choice(string.ascii_letters) for i in range(length))

    def get_random_alpha_numeric_string(self, length):
        return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

    def get_seed(self):
        return self.seed

    def random_choice(self, list):
        return random.choice(list)

    def get_random_float(self, low, high):
        return str(self.get_random_integer(low, high)) + "." + str(self.get_random_integer(1, 999))

    def shuffle_list(self, list):
        random.shuffle(list)

    def get_a_non_prime_integer(self, max):
        while True:
            number = self.get_random_integer(4, max)
            for i in range(2, number):
                if number % i == 0:
                    return number

    def get_random_alpha(self):
        return random.random()


# generated formula snippets
# simple rules

def generate_nodes(assignment, randomness, theory):
    generated_list = list()
    flag = [None] * 20000
    int_nodes = list()
    real_nodes = list()
    string_nodes = list()
    bv_nodes = list()
    temp_int = list()
    temp_real = list()
    temp_str = list()
    temp_bv = list()
    for i in range(len(assignment)):
        var = assignment[i]
        flag[i] = var
        if type(assignment[flag[i]]) == z3.z3.IntNumRef:
            int_nodes.append(flag[i])
        elif type(assignment[flag[i]]) == z3.z3.SeqRef:
            string_nodes.append(flag[i])
        elif type(assignment[flag[i]]) == z3.z3.RatNumRef:
            real_nodes.append(flag[i])
        elif type(assignment[flag[i]]) == z3.z3.BitVecNumRef:
            bv_nodes.append(flag[i])

    if len(int_nodes) != 0:
        for i in range(10):
            x_index = randomness.get_random_integer(0, len(int_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(int_nodes) - 1)
            if theory in ["AUFNIA", "AUFNIRA", "NIA", "NRA", "QF_ANIA", "QF_AUFNIA", "QF_NIA", "QF_NIRA", "QF_NRA",
                          "QF_UFNIA", "QF_UFNRA", "UFNIA", "UFNRA"]:
                operate = randomness.random_choice(["plus", "mul", "mul_and_plus"])
            elif theory in ["QF_IDL", "QF_RDL", "QF_UFIDL", "UFIDL"]:
                operate = "diff"
            else:
                operate = randomness.random_choice(["plus", "mul_and_plus"])
            x = Int("%s" % int_nodes[x_index])
            y = Int("%s" % int_nodes[y_index])
            x_value = assignment[x].as_long()
            y_value = assignment[y].as_long()
            if operate == "plus":
                has_constant = randomness.random_choice([True, False])
                if has_constant:
                    a = randomness.get_random_integer(1, 10000)
                    result = x_value + y_value + a
                    temp_int.append(x + y + a == result)
                else:
                    result = x_value + y_value
                    temp_int.append(x + y == result)
            elif operate == "mul":
                result = x_value * y_value
                temp_int.append(x * y == result)
            elif operate == "diff":
                a = randomness.get_random_integer(1, 10)
                b = randomness.get_random_integer(1, 10)
                max_result_x = x_value + a
                min_result_x = x_value - a
                max_result_y = y_value + b
                min_result_y = y_value - b
                temp_int.append(And(x < max_result_x, x > min_result_x, y < max_result_y, y > min_result_y))
            elif operate == "mul_and_plus":
                a = randomness.get_random_integer(1, 100)
                b = randomness.get_random_integer(1, 100)
                c = randomness.get_random_integer(1, 10000)
                result = a * x_value + b * y_value + c
                temp_int.append(a * x + b * y + c == result)

    if len(real_nodes) != 0:
        for i in range(10):
            x_index = randomness.get_random_integer(0, len(real_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(real_nodes) - 1)
            if theory in ["AUFNIA", "AUFNIRA", "NIA", "NRA", "QF_ANIA", "QF_AUFNIA", "QF_NIA", "QF_NIRA", "QF_NRA",
                          "QF_UFNIA", "QF_UFNRA", "UFNIA", "UFNRA"]:
                operate = randomness.random_choice(["plus", "mul", "mul_and_plus"])
            elif theory in ["QF_IDL", "QF_RDL", "QF_UFIDL", "UFIDL"]:
                operate = "diff"
            else:
                operate = randomness.random_choice(["plus", "mul_and_plus"])
            x = Real("%s" % real_nodes[x_index])
            y = Real("%s" % real_nodes[y_index])
            x_value = assignment[x]
            y_value = assignment[y]

            if operate == "plus":
                has_constant = randomness.random_choice([True, False])
                if has_constant:
                    a = float(randomness.get_random_float(1, 10000))
                    result = x_value + y_value + a
                    temp_real.append(x + y + a == result)
                else:
                    result = x_value + y_value
                    temp_real.append(x + y == result)
            elif operate == "mul":
                result = x_value * y_value
                temp_real.append(x * y == result)
            elif operate == "diff":
                a = float(randomness.get_random_float(0, 2))
                b = float(randomness.get_random_float(0, 2))
                max_result_x = x_value + a
                min_result_x = x_value - a
                max_result_y = y_value + b
                min_result_y = y_value - b
                temp_real.append(And(x < max_result_x, x > min_result_x, y < max_result_y, y > min_result_y))
            elif operate == "mul_and_plus":
                a = float(randomness.get_random_float(0, 100))
                b = float(randomness.get_random_float(0, 100))
                c = float(randomness.get_random_float(0, 10000))
                result = a * x_value + b * y_value + c
                temp_real.append(a * x + b * y + c == result)

    if len(string_nodes) != 0:
        for i in range(10):
            x_index = randomness.get_random_integer(0, len(string_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(string_nodes) - 1)
            x = String("%s" % string_nodes[x_index])
            y = String("%s" % string_nodes[y_index])
            x_value = assignment[x].as_string()
            y_value = assignment[y].as_string()
            has_constant = randomness.random_choice([True, False])
            if has_constant:
                a = randomness.get_random_alpha_numeric_string(randomness.get_random_integer(1, 20))
                result = StringVal(x_value + a + y_value)
                temp_str.append(x + a + y == result)
            else:
                result = StringVal(x_value + y_value)
                temp_str.append(x + y == result)

    if len(bv_nodes) != 0:
        for i in range(10):
            x_index = randomness.get_random_integer(0, len(bv_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(bv_nodes) - 1)
            chosen_num = 0
            size = assignment[bv_nodes[x_index]].size()
            while assignment[bv_nodes[y_index]].size() != size and chosen_num < 5:
                y_index = randomness.get_random_integer(0, len(bv_nodes) - 1)
            if assignment[bv_nodes[y_index]].size() == size:
                x = BitVec("%s" % bv_nodes[x_index], size)
                y = BitVec("%s" % bv_nodes[y_index], size)
                x_value = assignment[x]
                y_value = assignment[y]
                a = randomness.get_random_integer(1, 100)
                add_res = x_value.as_long() + y_value.as_long() + a
                if add_res < (2 ** size) - 1:
                    result = BitVecVal(add_res, size)
                    constant = BitVecVal(a, size)
                    temp_bv.append(x + constant + y == result)
                elif x_value.as_long() - y_value.as_long() > 0:
                    sub_res = x_value.as_long() - y_value.as_long() + a
                    result = BitVecVal(sub_res, size)
                    constant = BitVecVal(a, size)
                    temp_bv.append(x + constant - y == result)
                elif x_value.as_long() - y_value.as_long() < 0:
                    sub_res = y_value.as_long() - x_value.as_long() + a
                    result = BitVecVal(sub_res, size)
                    constant = BitVecVal(a, size)
                    temp_bv.append(y + constant - x == result)

    if len(int_nodes) == 0 and len(real_nodes) == 0 and len(string_nodes) == 0 and len(bv_nodes) == 0:
        generated_list = None
    else:
        generated_list = temp_str + temp_real + temp_int + temp_bv
    return generated_list


def complete_generate_nodes(assignment, randomness, theory):
    # global exp2, exp1, v1, v2

    def generate_comparison_operator(tp, comparision_operator, expression, value):
        if tp == "int":
            if comparision_operator == "=":
                temp_int.append(expression == value)
            elif comparision_operator == ">":
                m = randomness.get_random_integer(1, 5)
                temp_int.append(expression > value - m)
            elif comparision_operator == ">=":
                m = randomness.get_random_integer(0, 3)
                temp_int.append(expression >= value - m)
            elif comparision_operator == "<":
                m = randomness.get_random_integer(1, 5)
                temp_int.append(expression < value + m)
            elif comparision_operator == "<=":
                m = randomness.get_random_integer(0, 3)
                temp_int.append(expression <= value + m)
        elif tp == "real":
            m = float(randomness.get_random_float(0, 2))
            if comparision_operator == "=":
                temp_real.append(expression == value)
            elif comparision_operator == ">":
                temp_real.append(expression > value - m)
            elif comparision_operator == ">=":
                temp_real.append(expression >= value - m)
            elif comparision_operator == "<":
                temp_real.append(expression < value + m)
            elif comparision_operator == "<=":
                temp_real.append(expression <= value + m)
        elif tp == "fp":
            m = randomness.get_random_integer(1, 8)
            if comparision_operator == "=":
                temp_fp.append(expression == value)
            elif comparision_operator == ">":
                temp_fp.append(expression > value - m)
            elif comparision_operator == ">=":
                temp_fp.append(expression >= value - m)
            elif comparision_operator == "<":
                temp_fp.append(expression < value + m)
            elif comparision_operator == "<=":
                temp_fp.append(expression <= value + m)

    generated_list = list()
    flag = [None] * 20000
    int_nodes = list()
    real_nodes = list()
    string_nodes = list()
    fp_nodes = list()
    bv_nodes = list()
    temp_int = list()
    temp_real = list()
    temp_str = list()
    temp_bv = list()
    temp_fp = list()
    for i in range(len(assignment)):
        var = assignment[i]
        flag[i] = var
        if type(assignment[flag[i]]) == z3.z3.IntNumRef:
            int_nodes.append(flag[i])
        elif type(assignment[flag[i]]) == z3.z3.SeqRef:
            string_nodes.append(flag[i])
        elif type(assignment[flag[i]]) == z3.z3.RatNumRef:
            real_nodes.append(flag[i])
        elif type(assignment[flag[i]]) == z3.z3.BitVecNumRef:
            bv_nodes.append(flag[i])
        elif type(assignment[flag[i]]) == z3.z3.FPNumRef:
            fp_nodes.append(flag[i])
            
    times = 10
    if len(int_nodes) != 0:
        for i in range(times):
            x_index = randomness.get_random_integer(0, len(int_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(int_nodes) - 1)
            x = Int("%s" % int_nodes[x_index])
            y = Int("%s" % int_nodes[y_index])
            x_value = assignment[x].as_long()
            y_value = assignment[y].as_long()
            if "LI" in theory:
                op1 = randomness.random_choice(["<", "<=", ">", ">=", "=", "=", "=", "="])
                op2 = randomness.random_choice(["+", "-"])
                if x_value > 0 and y_value > 0:
                    op3 = randomness.random_choice(["*", "/", "%"])
                else:
                    op3 = "*"
                a = randomness.get_random_integer(1, 10000)
                b = randomness.get_random_integer(1, 10000)
                c = randomness.get_random_integer(1, 10000)
                if op3 == "*":
                    exp1 = a * x
                    exp2 = b * y
                    v1 = a * x_value
                    v2 = b * y_value
                elif op3 == "/":
                    exp1 = x / a
                    exp2 = y / b
                    v1 = x_value // a
                    v2 = y_value // b
                else:
                    exp1 = x % a
                    exp2 = y % b
                    v1 = x_value % a
                    v2 = y_value % b

                if op2 == "+":
                    exp = exp1 + c + exp2
                    v = v1 + v2 + c
                else:
                    exp = exp1 - c - exp2
                    v = v1 - v2 - c
                generate_comparison_operator("int", op1, exp, v)

            elif "NI" in theory:
                op1 = randomness.random_choice(["<", "<=", ">", ">=", "=", "=", "=", "="])
                if x_value > 0:
                    op2 = randomness.random_choice(["*", "/", "%"])
                else:
                    op2 = "*"
                if y_value > 0:
                    op3 = randomness.random_choice(["*", "/", "%"])
                else:
                    op3 = "*"
                c = randomness.get_random_integer(1, 10000)
                if op2 == "*":
                    exp1 = c * x
                    v1 = c * x_value
                elif op2 == "/":
                    exp1 = c / x
                    v1 = c // x_value
                else:
                    exp1 = c % x
                    v1 = c % x_value
                if op3 == "*":
                    exp = exp1 * y
                    v = v1 * y_value
                elif op3 == "/":
                    exp = exp1 / y
                    v = v1 // y_value
                else:
                    exp = exp1 % y
                    v = v1 % y_value
                generate_comparison_operator("int", op1, exp, v)

            elif "IDL" in theory:
                op1 = randomness.random_choice(["<", "<=", ">", ">=", "="])
                exp = x - y
                v = x_value - y_value
                generate_comparison_operator("int", op1, exp, v)

    if len(real_nodes) != 0:
        for i in range(times):
            x_index = randomness.get_random_integer(0, len(real_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(real_nodes) - 1)
            x = Real("%s" % real_nodes[x_index])
            y = Real("%s" % real_nodes[y_index])
            x_value = assignment[x]
            y_value = assignment[y]
            if "L" in theory and "R" in theory:
                op1 = randomness.random_choice(["<", "<=", ">", ">=", "=", "=", "=", "="])
                op2 = randomness.random_choice(["+", "-"])
                op3 = randomness.random_choice(["*", "/"])
                a = float(randomness.get_random_float(1, 10000))
                b = float(randomness.get_random_float(1, 10000))
                c = float(randomness.get_random_float(1, 10000))
                if op3 == "*":
                    exp1 = a * x
                    exp2 = b * y
                    v1 = a * x_value
                    v2 = b * y_value
                else:
                    exp1 = x / a
                    exp2 = y / b
                    v1 = x_value / a
                    v2 = y_value / b
                if op2 == "+":
                    exp = exp1 + c + exp2
                    v = v1 + v2 + c
                else:
                    exp = exp1 - c - exp2
                    v = v1 - v2 - c
                generate_comparison_operator("real", op1, exp, v)

            elif "N" in theory and "R" in theory:
                op1 = randomness.random_choice(["<", "<=", ">", ">=", "=", "=", "=", "="])
                if x_value.as_string() != "0" or x_value.as_string() != "0/0":
                    op2 = randomness.random_choice(["*", "/"])
                else:
                    op2 = "*"
                if y_value.as_string() != "0" or y_value.as_string() != "0/0":
                    op3 = randomness.random_choice(["*", "/"])
                else:
                    op3 = "*"
                c = float(randomness.get_random_float(1, 10000))
                if op2 == "*":
                    exp1 = c * x
                    v1 = c * x_value
                else:
                    exp1 = c / x
                    v1 = c / x_value
                if op3 == "*":
                    exp = exp1 * y
                    v = v1 * y_value
                else:
                    exp = exp1 / y
                    v = v1 / y_value
                generate_comparison_operator("real", op1, exp, v)

            elif "RDL":
                op1 = randomness.random_choice(["<", "<=", ">", ">=", "="])
                exp = x - y
                v = x_value - y_value
                generate_comparison_operator("real", op1, exp, v)

    if len(string_nodes) != 0:
        for i in range(times):
            x_index = randomness.get_random_integer(0, len(string_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(string_nodes) - 1)
            x = String("%s" % string_nodes[x_index])
            y = String("%s" % string_nodes[y_index])
            x_value = assignment[x].as_string()
            y_value = assignment[y].as_string()
            # has_constant = randomness.random_choice([True, False])
            op = randomness.random_choice(["+", "Contains"])
            a = randomness.get_random_alpha_numeric_string(randomness.get_random_integer(1, 20))
            result = StringVal(x_value + a + y_value)
            if op == "+":
                temp_str.append(x + a + y == result)
            else:
                temp_str.append(Contains(result, x))

    if len(bv_nodes) != 0:
        for i in range(times):
            x_index = randomness.get_random_integer(0, len(bv_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(bv_nodes) - 1)
            chosen_num = 0
            size = assignment[bv_nodes[x_index]].size()
            while assignment[bv_nodes[y_index]].size() != size and chosen_num < 5:
                y_index = randomness.get_random_integer(0, len(bv_nodes) - 1)
                chosen_num = chosen_num + 1
            if assignment[bv_nodes[y_index]].size() == size:
                x = BitVec("%s" % bv_nodes[x_index], size)
                y = BitVec("%s" % bv_nodes[y_index], size)
                x_value = assignment[x]
                y_value = assignment[y]
                op = randomness.random_choice(["+", "&", "*", "/"])
                a = randomness.get_random_integer(1, 100)
                v_x = BitVecVal(x_value.as_long(), size)
                v_y = BitVecVal(y_value.as_long(), size)
                if op == "+":
                    add_res = x_value.as_long() + y_value.as_long() + a
                    if add_res < (2 ** size) - 1:
                        result = BitVecVal(add_res, size)
                        constant = BitVecVal(a, size)
                        temp_bv.append(x + constant + y == result)
                    elif x_value.as_long() - y_value.as_long() > 0:
                        sub_res = x_value.as_long() - y_value.as_long() + a
                        result = BitVecVal(sub_res, size)
                        constant = BitVecVal(a, size)
                        temp_bv.append(x + constant - y == result)
                    elif x_value.as_long() - y_value.as_long() < 0:
                        sub_res = y_value.as_long() - x_value.as_long() + a
                        result = BitVecVal(sub_res, size)
                        constant = BitVecVal(a, size)
                        temp_bv.append(y + constant - x == result)
                elif op == "&":
                    value = v_x & v_y
                    temp_bv.append(y & x == value)
                elif op == "*":
                    value = v_x * v_y
                    temp_bv.append(x * y == value)
                elif op == "/":
                    value = v_x / v_y
                    temp_bv.append(x / y == value)

    if len(fp_nodes) != 0:
        for i in range(times):
            x_index = randomness.get_random_integer(0, len(fp_nodes) - 1)
            y_index = randomness.get_random_integer(0, len(fp_nodes) - 1)
            chosen_num = 0
            sort = assignment[fp_nodes[x_index]].sort()
            while assignment[fp_nodes[y_index]].sort() != sort and chosen_num < 5:
                y_index = randomness.get_random_integer(0, len(fp_nodes) - 1)
                chosen_num = chosen_num + 1
            if assignment[fp_nodes[y_index]].sort() == sort:
                x = FP("%s" % fp_nodes[x_index], sort)
                y = FP("%s" % fp_nodes[y_index], sort)
                x_value = assignment[x]
                y_value = assignment[y]
                if x_value is not None and y_value is not None:
                    op1 = randomness.random_choice(["+", "-", "*", "/"])
                    op3 = "="
                    if op1 == "+":
                        exp = y + x
                        v = y_value + x_value
                    elif op1 == "-":
                        exp = y - x
                        v = y_value - x_value
                    elif op1 == "*":
                        exp = y * x
                        v = y_value * x_value
                    else:
                        exp = x / y
                        v = x_value / y_value
                    generate_comparison_operator("fp", op3, exp, v)

    if len(int_nodes) == 0 and len(real_nodes) == 0 and len(string_nodes) == 0 and len(
            bv_nodes) == 0 and len(fp_nodes) == 0:
        generated_list = None
    else:
        generated_list = temp_str + temp_real + temp_int + temp_bv + temp_fp
    return generated_list


def gen_formula_of_logic(logic):
    randomness = Randomness(random.randint(0, 15))
    if "I" in logic:
        w, x, y, z = Ints("w x y z")
    elif "R" in logic:
        w, x, y, z = Reals("w x y z")
    elif "BV" in logic:
        w, x, y, z = BitVecs("w x y z", 16)
    elif "S" in logic:
        w, x, y, z = Strings("w x y z")
    elif "FP" in logic:
        w, x, y, z = FPs("w x y z", FPSort(8, 24))
    else:
        w, x, y, z = Ints("w x y z")
    fml = And(w == x, x == y, y == z)
    s = Solver()
    s.add(fml)
    s.check()
    m = s.model()
    # if random.random() < 0.5:
    atoms = complete_generate_nodes(m, randomness, logic)
    # else: atoms = generate_nodes(m, randomness, logic)

    max_assert = random.randint(3, 20)
    res = []
    assert (len(atoms) >= 1)
    for _ in range(max_assert):
        clen = random.randint(1, 8)  # clause length
        if clen == 1:
            cls = random.choice(atoms)
        else:
            cls = Or(random.sample(atoms, min(len(atoms), clen)))
        res.append(cls)
    if len(res) == 1:
        return res[0]
    else:
        return And(res)


if __name__ == "__main__":
    print(gen_formula_of_logic("QF_BV"))
