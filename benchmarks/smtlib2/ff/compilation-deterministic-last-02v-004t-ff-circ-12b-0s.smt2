(set-info :smt-lib-version 2.6)
(set-info :category "crafted")
(set-info :status "sat")
(set-logic QF_FFA)
(define-sort FF0 () (_ FiniteField 2053))
(declare-fun a_n2_alt () FF0)
(declare-fun b_n1_alt () FF0)
(declare-fun is_zero_n4_alt () FF0)
(declare-fun is_zero_inv_n3_alt () FF0)
(declare-fun a_n2 () FF0)
(declare-fun b_n1 () FF0)
(declare-fun is_zero_n4 () FF0)
(declare-fun is_zero_inv_n3 () FF0)
(declare-fun return_n0_alt () FF0)
(declare-fun return_n0 () FF0)
(assert 
  (let ((let0 (as ff0 FF0)))
  (let ((let1 (as ff5 FF0)))
  (let ((let2 (as ff2050 FF0)))
  (let ((let3 a_n2_alt))
  (let ((let4 (ff.mul let3 let2)))
  (let ((let5 (as ff2051 FF0)))
  (let ((let6 b_n1_alt))
  (let ((let7 (ff.mul let6 let5)))
  (let ((let8 (ff.add let7 let4 let1)))
  (let ((let9 is_zero_n4_alt))
  (let ((let10 (ff.mul let9 let8)))
  (let ((let11 (= let10 let0)))
  (let ((let12 (as ff1 FF0)))
  (let ((let13 (as ff2052 FF0)))
  (let ((let14 (ff.mul let9 let13)))
  (let ((let15 (ff.add let14 let12)))
  (let ((let16 is_zero_inv_n3_alt))
  (let ((let17 (ff.mul let16 let8)))
  (let ((let18 (= let17 let15)))
  (let ((let19 (and let18 let11)))
  (let ((let20 a_n2))
  (let ((let21 (ff.mul let20 let2)))
  (let ((let22 b_n1))
  (let ((let23 (ff.mul let22 let5)))
  (let ((let24 (ff.add let23 let21 let1)))
  (let ((let25 is_zero_n4))
  (let ((let26 (ff.mul let25 let24)))
  (let ((let27 (= let26 let0)))
  (let ((let28 (ff.mul let25 let13)))
  (let ((let29 (ff.add let28 let12)))
  (let ((let30 is_zero_inv_n3))
  (let ((let31 (ff.mul let30 let24)))
  (let ((let32 (= let31 let29)))
  (let ((let33 (and let32 let27)))
  (let ((let34 return_n0_alt))
  (let ((let35 return_n0))
  (let ((let36 (= let35 let34)))
  (let ((let37 (not let36)))
  (let ((let38 (= let20 let3)))
  (let ((let39 (= let22 let6)))
  (let ((let40 (and let39 let38)))
  (let ((let41 (and let40 let37 let33 let19)))
  let41
))))))))))))))))))))))))))))))))))))))))))
)
(check-sat)
