(set-info :smt-lib-version 2.6)
(set-logic QF_FFA)
(set-info :status "unsat")
(define-sort FF0 () (_ FiniteField 3))
(declare-fun x0 () FF0)
(declare-fun x1 () FF0)
(declare-fun x2 () FF0)
(declare-fun x3 () FF0)
(declare-fun x4 () FF0)
(declare-fun x5 () FF0)
(declare-fun x6 () FF0)
(declare-fun x7 () FF0)
(assert
  (let ((let0 (ff.mul (as ff2 FF0) x1 x1 x2)))
  (let ((let1 (ff.mul x0 x4 x7)))
  (let ((let2 (ff.mul x1 x6 x7)))
  (let ((let3 (ff.mul x5 x6 x7)))
  (let ((let4 (as ff1 FF0)))
  (let ((let5 (ff.add let0 let1 let2 let3 let4)))
  (let ((let6 (= let5 (as ff0 FF0))))
  (let ((let7 (ff.mul x0 x0 x3)))
  (let ((let8 (ff.mul x0 x6 x6)))
  (let ((let9 x4))
  (let ((let10 (ff.add let7 let8 let9)))
  (let ((let11 (= let10 (as ff0 FF0))))
  (let ((let12 (ff.mul x3 x4 x4)))
  (let ((let13 (ff.mul x0 x3 x6)))
  (let ((let14 (ff.mul x6 x7 x7)))
  (let ((let15 (ff.add let12 let13 let14)))
  (let ((let16 (= let15 (as ff0 FF0))))
  (let ((let17 (ff.mul x0 x1 x6)))
  (let ((let18 (ff.mul x2 x6 x6)))
  (let ((let19 (ff.mul (as ff2 FF0) x4 x7 x7)))
  (let ((let20 (ff.add let17 let18 let19)))
  (let ((let21 (= let20 (as ff0 FF0))))
  (let ((let22 (ff.mul (as ff2 FF0) x1 x3 x4)))
  (let ((let23 (ff.mul x1 x4)))
  (let ((let24 (ff.add let22 let23)))
  (let ((let25 (= let24 (as ff0 FF0))))
  (let ((let26 (ff.mul (as ff2 FF0) x1 x5 x7)))
  (let ((let27 x3))
  (let ((let28 (ff.add let26 let27)))
  (let ((let29 (= let28 (as ff0 FF0))))
  (let ((let30 (ff.mul x2 x3 x3)))
  (let ((let31 (ff.mul (as ff2 FF0) x1 x3 x6)))
  (let ((let32 (ff.mul (as ff2 FF0) x5 x7 x7)))
  (let ((let33 (ff.add let30 let31 let32)))
  (let ((let34 (= let33 (as ff0 FF0))))
  (let ((let35 (ff.mul (as ff2 FF0) x1 x6 x7)))
  (let ((let36 (ff.mul (as ff2 FF0) x0 x3)))
  (let ((let37 x1))
  (let ((let38 (ff.add let35 let36 let37)))
  (let ((let39 (= let38 (as ff0 FF0))))
  (let ((let40 (and let6 let11 let16 let21 let25 let29 let34 let39)))
  let40
)))))))))))))))))))))))))))))))))))))))))
)
(check-sat)
