(set-info :smt-lib-version 2.6)
(set-logic QF_FF)
(set-info :status 'sat')
(define-sort FF0 () (_ FiniteField 52435875175126190479447740508185965837690552500527637822603658699938581184513))
(declare-fun _2_alt () FF0)
(declare-fun _1_alt () FF0)
(declare-fun _0_alt () FF0)
(declare-fun _2 () FF0)
(declare-fun _1 () FF0)
(declare-fun _0 () FF0)
(declare-fun out_alt () FF0)
(declare-fun out () FF0)
(assert 
  (let ((let0 _2_alt))
  (let ((let1 _1_alt))
  (let ((let2 _0_alt))
  (let ((let3 (ff.mul let2 let1)))
  (let ((let4 (= let3 let0)))
  (let ((let5 (ff.mul let1 let1)))
  (let ((let6 (= let5 let1)))
  (let ((let7 (ff.mul let2 let2)))
  (let ((let8 (= let7 let2)))
  (let ((let9 (and let8 let6 let4)))
  (let ((let10 _2))
  (let ((let11 _1))
  (let ((let12 _0))
  (let ((let13 (ff.mul let12 let11)))
  (let ((let14 (= let13 let10)))
  (let ((let15 (ff.mul let11 let11)))
  (let ((let16 (= let15 let11)))
  (let ((let17 (ff.mul let12 let12)))
  (let ((let18 (= let17 let12)))
  (let ((let19 (and let18 let16 let14)))
  (let ((let20 out_alt))
  (let ((let21 out))
  (let ((let22 (= let21 let20)))
  (let ((let23 (not let22)))
  (let ((let24 (= let12 let2)))
  (let ((let25 (= let11 let1)))
  (let ((let26 (and let25 let24)))
  (let ((let27 (and let26 let23 let19 let9)))
  let27
))))))))))))))))))))))))))))
)
(check-sat)
