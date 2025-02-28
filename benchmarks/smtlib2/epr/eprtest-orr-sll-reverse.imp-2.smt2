(set-option :print-success false)
(set-option :produce-proofs false)

(set-logic UF)
(set-info :source |First push block from orr-sanitized-eeaa/sll-reverse.imp.smt2|)
(set-info :smt-lib-version 2.0)
(set-info :category "crafted")
(set-info :status unsat)

(declare-sort V 0)
(declare-fun n* (V V) Bool)
(declare-fun i () V)
(declare-fun h () V)
(declare-fun k () V)
(declare-fun j () V)
(declare-fun _n* (V V) Bool)
(declare-fun null () V)

(declare-fun EQ (V V) Bool)
(assert (forall ((x V)) (EQ x x)))
(assert (forall ((x V) (y V)) (=> (EQ x y) (EQ y x))))
(assert (forall ((x V) (y V) (z V)) (=> (and (EQ x y) (EQ y z)) (EQ x z))))
(assert (forall ((x0 V) (y0 V) (x1 V) (y1 V)) (=> (and (EQ x0 y0) (EQ x1 y1)) (=> (_n* x0 x1) (_n* y0 y1)))))
(assert (forall ((x0 V) (y0 V) (x1 V) (y1 V)) (=> (and (EQ x0 y0) (EQ x1 y1)) (=> (n* x0 x1) (n* y0 y1)))))

(assert (forall ((u$1$1 V)) (n* u$1$1 u$1$1)))
(assert (forall ((u$2$1 V) (v$1$1 V) (w$1$1 V)) (=> (and (n* u$2$1 v$1$1) (n* v$1$1 w$1$1)) (n* u$2$1 w$1$1))))
(assert (forall ((u$3$1 V) (v$2$1 V) (w$2$1 V)) (=> (and (n* u$3$1 v$2$1) (n* u$3$1 w$2$1)) (or (n* v$2$1 w$2$1) (n* w$2$1 v$2$1)))))
(assert (forall ((u$4$1 V) (v$3$1 V)) (=> (n* u$4$1 v$3$1) (=> (n* v$3$1 u$4$1) (EQ u$4$1 v$3$1)))))
(assert (forall ((v$4$1 V)) (=> (or (n* null v$4$1) (n* v$4$1 null)) (EQ null v$4$1))))
(assert (forall ((u$5$1 V)) (_n* u$5$1 u$5$1)))
(assert (forall ((u$6$1 V) (v$5$1 V) (w$3$1 V)) (=> (and (_n* u$6$1 v$5$1) (_n* v$5$1 w$3$1)) (_n* u$6$1 w$3$1))))
(assert (forall ((u$7$1 V) (v$6$1 V) (w$4$1 V)) (=> (and (_n* u$7$1 v$6$1) (_n* u$7$1 w$4$1)) (or (_n* v$6$1 w$4$1) (_n* w$4$1 v$6$1)))))
(assert (forall ((u$8$1 V) (v$7$1 V)) (=> (_n* u$8$1 v$7$1) (=> (_n* v$7$1 u$8$1) (EQ u$8$1 v$7$1)))))
(assert (forall ((v$8$1 V)) (=> (or (_n* null v$8$1) (_n* v$8$1 null)) (EQ null v$8$1))))

(assert (not (=> (and (forall ((u$18$1 V) (v$14$1 V)) (= (_n* u$18$1 v$14$1) (n* u$18$1 v$14$1))) (forall ((u$19$1 V)) (=> (not (EQ u$19$1 null)) (n* h u$19$1)))) (and (forall ((u$20$1 V)) (=> (not (EQ u$20$1 null)) (= (n* h u$20$1) (not (n* null u$20$1))))) (forall ((u$21$1 V)) (=> (not (EQ u$21$1 null)) (_n* h u$21$1))) (forall ((u$22$1 V) (v$15$1 V)) (=> (n* h u$22$1) (= (n* u$22$1 v$15$1) (_n* u$22$1 v$15$1)))) (forall ((u$23$1 V) (v$16$1 V)) (=> (n* null u$23$1) (= (n* u$23$1 v$16$1) (_n* v$16$1 u$23$1)))) true))))

(check-sat)

(exit)
