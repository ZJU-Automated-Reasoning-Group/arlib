from machsmt import Benchmark,args
import json

if __name__ == '__main__':
    print(args.benchmark)
    benchmark = Benchmark(args.benchmark)
    benchmark.parse()
    feature = benchmark.get_features()
    dataset = args.dataset
    benchmarkname = args.benchmark.replace("../data/",'')
    benchmarkname = benchmarkname.replace("/","_")
    fea = {}
    fea[benchmarkname] = feature
    print(feature)
    print(str(benchmarkname) + ".json")