; strong concat
(set-logic SLIA)
(synth-fun f ((x String) (y String)) String
  ((Start String) (ntString String))
  ((Start String (ntString
                  (str.++ ntString ntString)))
   (ntString String (x y))))

(constraint (= (f "hello" "world") "helloworld"))
(constraint (= (f "syn" "thesis") "synthesis"))
(constraint (= (f "" "empty") "empty"))

(check-synth)