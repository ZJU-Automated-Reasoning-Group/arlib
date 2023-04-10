from pysat.formula import CNF
import sys


def cnf2lp(inf=None, outf=None):
    if inf == None:
        return
    f = CNF(inf)
    if outf != None:
        wf = open(outf, "w")
    else:
        wf = sys.stdout
    for c in f.clauses:
        head = ' | '.join(['p' + str(x) for x in c if x > 0])
        body = ', '.join(['p' + str(-x) for x in c if x < 0])
        if body != '':
            head = head + " :- "
        print(head + body + '.', file=wf)
    wf.close()


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("usage %s INPUT_CNF [OUTPUT_LP]" % sys.argv[0])
        sys.exit()
    if len(sys.argv) == 2:
        cnf2lp(sys.argv[1])
    else:
        cnf2lp(sys.argv[1], sys.argv[2])
