; Rotate bits right by 1
(set-logic BV)
(synth-fun f ((x (_ BitVec 4))) (_ BitVec 4))

(constraint (= (f #x1) #x8))
(constraint (= (f #x2) #x1))
(constraint (= (f #x4) #x2))
(constraint (= (f #x8) #x4))
(constraint (= (f #xF) #xF))
(constraint (= (f #x0) #x0))

(check-synth)