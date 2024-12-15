; Find least significant set bit

(set-logic BV)
(synth-fun f ((x (_ BitVec 4))) (_ BitVec 4))

(constraint (= (f #x0) #x0))
(constraint (= (f #x1) #x1))
(constraint (= (f #x2) #x2))
(constraint (= (f #x3) #x1))
(constraint (= (f #x4) #x4))
(constraint (= (f #x7) #x1))
(constraint (= (f #x8) #x8))
(constraint (= (f #xF) #x1))

(check-synth)