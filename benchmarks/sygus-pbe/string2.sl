; Extract first 3 characters
(set-logic SLIA)
(synth-fun f ((x String)) String
    ((Start String) (StartInt Int))
    ((Start String (x
                   (str.substr x StartInt StartInt)))
     (StartInt Int (0 1 2 3))))

(constraint (= (f "hello") "hel"))
(constraint (= (f "synthesis") "syn"))
(constraint (= (f "programs") "pro"))

(check-synth)