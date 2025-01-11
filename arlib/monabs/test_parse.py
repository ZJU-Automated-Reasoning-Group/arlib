bv_one = """
;; Example 1: State transition verification with QF_BV
(set-logic QF_BV)
(declare-const state (_ BitVec 8))
(declare-const input (_ BitVec 8))
(declare-const output (_ BitVec 8))

;; Initial state constraints
(assert (= state #x00))
(assert (bvult input #x10))

;; First transition
(push 1)
(assert (= output (bvadd state input)))
(assert (= state output))
(check-sat)
(get-model)

;; Second transition, new constraints
(push 1)
(assert (bvuge input #x08))
(assert (= output (bvmul state #x02)))
(check-sat)
(get-model)
(pop 1)

;; Alternative second transition
(assert (bvult input #x08))
(assert (= output (bvdiv state #x02)))
(check-sat)
(get-model)
(pop 1)
"""

abv_one = """
;; Example 2: Memory operations with QF_ABV
(set-logic QF_ABV)
(declare-const mem (Array (_ BitVec 16) (_ BitVec 8)))
(declare-const addr (_ BitVec 16))
(declare-const val (_ BitVec 8))

;; Base memory state
(assert (bvult addr #x1000))
(assert (= (select mem addr) val))

;; Write sequence
(push 1)
(assert (= mem (store mem addr #xFF)))
(assert (bvult val #x80))

(push 1)
;; Write to next address
(assert (= mem (store mem (bvadd addr #x0001) #xAA)))
(check-sat)
(get-model)
(pop 1)

;; Alternative: write to previous address
(assert (= mem (store mem (bvsub addr #x0001) #x55)))
(check-sat)
(get-model)
(pop 1)
"""

bv_two = """
;; Example 3: Protocol state machine with QF_BV
(set-logic QF_BV)
(declare-const state (_ BitVec 4))
(declare-const flags (_ BitVec 8))
(declare-const counter (_ BitVec 8))

;; Initial protocol state
(assert (= state #x0))
(assert (= flags #x00))

;; First state transition
(push 1)
(assert (= state #x1))
(assert (= flags (bvor flags #x01)))  ; Set first flag
(assert (= counter #x00))

(push 1)
;; Success path
(assert (bvult counter #x0A))
(assert (= flags (bvor flags #x02)))  ; Set second flag
(check-sat)
(get-model)
(pop 1)

;; Error path
(assert (bvuge counter #x0A))
(assert (= flags (bvor flags #x80)))  ; Set error flag
(check-sat)
(get-model)
(pop 1)
"""

abv_two = """
;; Example 4: Hardware verification with QF_ABV
(set-logic QF_ABV)
(declare-const cache (Array (_ BitVec 8) (_ BitVec 32)))
(declare-const memory (Array (_ BitVec 8) (_ BitVec 32)))
(declare-const addr (_ BitVec 8))
(declare-const data (_ BitVec 32))

;; Initial cache-memory consistency
(assert (= (select cache addr) (select memory addr)))

;; Cache write
(push 1)
(assert (= cache (store cache addr data)))
(assert (bvult data #x00000100))

(push 1)
;; Cache write-through
(assert (= memory (store memory addr data)))
(check-sat)
(get-model)
(pop 1)

;; Cache write-back
(assert (not (= (select cache addr) 
                (select memory addr))))
(check-sat)
(get-model)
(pop 1)
"""

bv_tree = """
;; Example 5: Bit manipulation with scoped constraints
(set-logic QF_BV)
(declare-const reg (_ BitVec 32))
(declare-const mask (_ BitVec 32))
(declare-const result (_ BitVec 32))

;; Base register state
(assert (= reg #x0000FF00))

(push 1)
;; Bit masking operation
(assert (= mask #x00FF0000))
(assert (= result (bvor reg (bvshl (bvand reg #x0000FF00) #x08))))

(push 1)
;; Additional constraint: upper bits must be zero
(assert (= (bvand result #xFF000000) #x00000000))
(check-sat)
(get-model)
(pop 1)

;; Alternative: set upper bits
(assert (= (bvand result #xFF000000) #xFF000000))
(check-sat)
(get-model)
(pop 1)
"""
