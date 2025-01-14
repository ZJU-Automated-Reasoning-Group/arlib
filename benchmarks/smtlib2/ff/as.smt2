; from CVC5
(set-info :smt-lib-version 2.6)
(set-info :category "crafted")
(set-logic QF_FF)
; all disjuncts should be false
(define-sort F () (_ FiniteField 17))
(assert (= (as ff0 F) (ff.add (as ff1 F) (ff.neg (as ff1 F)))))
(check-sat)