; EXPECT: sat
(set-logic QF_FF)
(set-info :smt-lib-version 2.6)
(set-info :category "crafted")
(declare-const v (_ FiniteField 3))
; v = v^2 + 2*(-1)
(assert (= v (ff.bitsum (ff.mul v v) (as ff-1 (_ FiniteField 3)))))
; 0 = v^2 - v - 2 = (v + 1)(v - 2)
; models are 1 and 2
(check-sat)
