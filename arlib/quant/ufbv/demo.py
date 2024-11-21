from z3 import *

fml_str = ''' \n
(assert \n
 (exists ((s (_ BitVec 5)) )(forall ((t (_ BitVec 5)) )(not (= (bvnand s t) (bvor s (bvneg t))))) \n
 ) \n
 ) \n
(check-sat)
'''


def split_list(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def demo_partition(max_bits, workers):
    over_apppro = []
    under_appro = []
    for i in range(workers):
        if i % 2 == 0:
            over_apppro.append(i)
        else:
            under_appro.append(i)

    # print("over, under: ", over_apppro, under_appro)

    max_bits_lists = []

    for i in range(1, max_bits + 1):
        max_bits_lists.append(i)

    print(max_bits_lists)
    print(split_list(max_bits_lists, len(over_apppro)))
    print(split_list(max_bits_lists, len(under_appro)))


def demo_solve_with_approx_partitioned(formula, reduction_type, q_type, bit_places, polarity, local_max_bit_width):
    """Recursively go through `formula` and approximate it. Check
    satisfiability of the approximated formula. Put the result to
    the `result_queue`.
    """
    print("appro solving start")
    print(formula)
    while (bit_places < (local_max_bit_width - 2) or
           max_bit_width == 0):

        # Approximate the formula
        approximated_formula = rec_go(formula,
                                      [],
                                      reduction_type,
                                      q_type,
                                      bit_places,
                                      polarity)

        s = z3.Tactic("ufbv").solver()
        s.add(approximated_formula)
        print(s.to_smt2())
        result = s.check()
        # pid = os.getpid()
        if q_type == Quantification.UNIVERSAL:
            logging.debug("over-appro solving finished")
        else:
            logging.debug("under-appro solving finished")

        if q_type == Quantification.UNIVERSAL:
            if (result == z3.CheckSatResult(z3.Z3_L_TRUE) or
                    result == z3.CheckSatResult(z3.Z3_L_UNDEF)):
                (reduction_type, bit_places) = next_approx(reduction_type,
                                                           bit_places)
            elif result == z3.CheckSatResult(z3.Z3_L_FALSE):
                logging.debug('over-appro success')
                return
        else:
            if result == z3.CheckSatResult(z3.Z3_L_TRUE):
                logging.debug('under-appro success')
                return

            elif (result == z3.CheckSatResult(z3.Z3_L_FALSE) or
                  result == z3.CheckSatResult(z3.Z3_L_UNDEF)):

                # Update reduction type and increase bit width
                (reduction_type, bit_places) = next_approx(reduction_type,
                                                           bit_places)

    solve_without_approx(formula)


def demo_main_multiple(formula):
    # Determine the type of reductio
    reduction_type = ReductionType.ONE_EXTENSION

    num_workers = 4

    # TODO: Not correct, should consider quant?
    # Get max bit
    # TODO: is the computed width correct???
    m_max_bit_width = extract_max_bits_for_formula(formula)
    partitioned_bits_lists = []
    logging.debug("max width: {}".format(m_max_bit_width))

    for i in range(1, m_max_bit_width + 1):
        partitioned_bits_lists.append(i)

    over_parts = split_list(partitioned_bits_lists, int(num_workers / 2))
    under_parts = split_list(partitioned_bits_lists, int(num_workers / 2))

    # logging.debug(over_parts)
    # logging.debug(under_parts)

    with multiprocessing.Manager() as manager:
        # result_queue = multiprocessing.Queue()

        # demo_process_queue = []

        for nth in range(num_workers):
            bits_id = int(nth / 2)
            if nth % 2 == 0:
                if len(over_parts[bits_id]) > 0:
                    start_width = over_parts[bits_id][0]
                    end_width = over_parts[bits_id][-1]
                else:
                    start_width = 1
                    end_width = m_max_bit_width
                # Over-approximation workers
                demo_solve_with_approx_partitioned(formula,
                                                   reduction_type,
                                                   Quantification.UNIVERSAL,
                                                   start_width,
                                                   Polarity.POSITIVE,
                                                   end_width)
            else:
                if len(over_parts[bits_id]) > 0:
                    start_width = under_parts[bits_id][0]
                    end_width = under_parts[bits_id][-1]
                else:
                    start_width = 1
                    end_width = m_max_bit_width
                # Under-approximation workers
                demo_solve_with_approx_partitioned(formula,
                                                   reduction_type,
                                                   Quantification.EXISTENTIAL,
                                                   start_width,
                                                   Polarity.POSITIVE,
                                                   end_width)


def demo_approximations():
    sol = Solver()
    sol.from_string(fml_str)
    fml = And(sol.assertions())
    demo_main_multiple(fml)


# demo_partition(32, 16)
# demo_partition(32, 16)
# demo_partition(7, 16)
# demo_partition(5, 16)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    demo_approximations()