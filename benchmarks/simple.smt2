(set-info :status unknown)
(declare-fun x () (_ BitVec 16))
(assert
 (let (($x16 (= x (_ bv6 16))))
 (let (($x17 (or (= x (_ bv5 16)) $x16)))
 (and (or (bvsgt x (_ bv10 16)) (= x (_ bv1 16))) $x17))))
(check-sat)