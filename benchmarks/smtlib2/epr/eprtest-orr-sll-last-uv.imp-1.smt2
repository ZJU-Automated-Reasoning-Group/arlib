(set-option :print-success false)
(set-option :produce-proofs false)
(set-logic UF)
(set-info :source |First push block from orr/sll-last.imp.smt2|)
(set-info :smt-lib-version 2.0)
(set-info :category "crafted")
(set-info :status unsat)

(declare-sort V 0)
(declare-fun a () V)
(declare-fun c () V)
(declare-fun b () V)
(declare-fun n* (V V) Bool)
(declare-fun q () V)
(declare-fun p () V)
(declare-fun u () V)
(declare-fun v () V)
(declare-fun null () V)

(declare-fun EQ (V V) Bool)
(assert (forall ((x V)) (EQ x x)))
(assert (forall ((x V) (y V)) (=> (EQ x y) (EQ y x))))
(assert (forall ((x V) (y V) (z V)) (=> (and (EQ x y) (EQ y z)) (EQ x z))))
(assert (forall ((x0 V) (y0 V) (x1 V) (y1 V)) (=> (and (EQ x0 y0) (EQ x1 y1)) (=> (n* x0 x1) (n* y0 y1)))))

(assert (forall ((u$1$1 V)) (n* u$1$1 u$1$1)))
(assert (forall ((u$2$1 V) (v$1$1 V) (w$1$1 V)) (=> (and (n* u$2$1 v$1$1) (n* v$1$1 w$1$1)) (n* u$2$1 w$1$1))))
(assert (forall ((u$3$1 V) (v$2$1 V) (w$2$1 V)) (=> (and (n* u$3$1 v$2$1) (n* u$3$1 w$2$1)) (or (n* v$2$1 w$2$1) (n* w$2$1 v$2$1)))))
(assert (forall ((u$4$1 V) (v$3$1 V)) (=> (n* u$4$1 v$3$1) (=> (n* v$3$1 u$4$1) (EQ u$4$1 v$3$1)))))
(assert (forall ((v$4$1 V)) (=> (or (n* null v$4$1) (n* v$4$1 null)) (EQ null v$4$1))))

(assert (not (=> (and (n* a c) (n* b c) (ite (EQ u null) (EQ p a) (and (n* a u) (or (and (n* u p) (not (EQ u p)) (forall ((w$3$1 V)) (=> (and (n* u w$3$1) (not (EQ u w$3$1))) (n* p w$3$1)))) (and (EQ p null) (forall ((w$4$1 V)) (not (and (n* u w$4$1) (not (EQ u w$4$1))))))))) (ite (EQ v null) (EQ q b) (and (n* b v) (or (and (n* v q) (not (EQ v q)) (forall ((w$5$1 V)) (=> (and (n* v w$5$1) (not (EQ v w$5$1))) (n* q w$5$1)))) (and (EQ q null) (forall ((w$6$1 V)) (not (and (n* v w$6$1) (not (EQ v w$6$1)))))))))) (ite (or (not (EQ p null)) (not (EQ q null))) (ite (not (EQ p null)) (and (not (EQ p null)) (forall ((z$1$1 V)) (=> (or (and (n* p z$1$1) (not (EQ p z$1$1)) (forall ((w$7$1 V)) (=> (and (n* p w$7$1) (not (EQ p w$7$1))) (n* z$1$1 w$7$1)))) (and (EQ z$1$1 null) (forall ((w$8$1 V)) (not (and (n* p w$8$1) (not (EQ p w$8$1))))))) (ite (not (EQ q null)) (and (not (EQ q null)) (forall ((z$2$1 V)) (=> (or (and (n* q z$2$1) (not (EQ q z$2$1)) (forall ((w$9$1 V)) (=> (and (n* q w$9$1) (not (EQ q w$9$1))) (n* z$2$1 w$9$1)))) (and (EQ z$2$1 null) (forall ((w$10$1 V)) (not (and (n* q w$10$1) (not (EQ q w$10$1))))))) (and (n* a c) (n* b c) (ite (EQ p null) (EQ z$1$1 a) (and (n* a p) (or (and (n* p z$1$1) (not (EQ p z$1$1)) (forall ((w$11$1 V)) (=> (and (n* p w$11$1) (not (EQ p w$11$1))) (n* z$1$1 w$11$1)))) (and (EQ z$1$1 null) (forall ((w$12$1 V)) (not (and (n* p w$12$1) (not (EQ p w$12$1))))))))) (ite (EQ q null) (EQ z$2$1 b) (and (n* b q) (or (and (n* q z$2$1) (not (EQ q z$2$1)) (forall ((w$13$1 V)) (=> (and (n* q w$13$1) (not (EQ q w$13$1))) (n* z$2$1 w$13$1)))) (and (EQ z$2$1 null) (forall ((w$14$1 V)) (not (and (n* q w$14$1) (not (EQ q w$14$1))))))))))))) (and (n* a c) (n* b c) (ite (EQ p null) (EQ z$1$1 a) (and (n* a p) (or (and (n* p z$1$1) (not (EQ p z$1$1)) (forall ((w$15$1 V)) (=> (and (n* p w$15$1) (not (EQ p w$15$1))) (n* z$1$1 w$15$1)))) (and (EQ z$1$1 null) (forall ((w$16$1 V)) (not (and (n* p w$16$1) (not (EQ p w$16$1))))))))) (ite (EQ v null) (EQ q b) (and (n* b v) (or (and (n* v q) (not (EQ v q)) (forall ((w$17$1 V)) (=> (and (n* v w$17$1) (not (EQ v w$17$1))) (n* q w$17$1)))) (and (EQ q null) (forall ((w$18$1 V)) (not (and (n* v w$18$1) (not (EQ v w$18$1)))))))))))))) (ite (not (EQ q null)) (and (not (EQ q null)) (forall ((z$3$1 V)) (=> (or (and (n* q z$3$1) (not (EQ q z$3$1)) (forall ((w$19$1 V)) (=> (and (n* q w$19$1) (not (EQ q w$19$1))) (n* z$3$1 w$19$1)))) (and (EQ z$3$1 null) (forall ((w$20$1 V)) (not (and (n* q w$20$1) (not (EQ q w$20$1))))))) (and (n* a c) (n* b c) (ite (EQ u null) (EQ p a) (and (n* a u) (or (and (n* u p) (not (EQ u p)) (forall ((w$21$1 V)) (=> (and (n* u w$21$1) (not (EQ u w$21$1))) (n* p w$21$1)))) (and (EQ p null) (forall ((w$22$1 V)) (not (and (n* u w$22$1) (not (EQ u w$22$1))))))))) (ite (EQ q null) (EQ z$3$1 b) (and (n* b q) (or (and (n* q z$3$1) (not (EQ q z$3$1)) (forall ((w$23$1 V)) (=> (and (n* q w$23$1) (not (EQ q w$23$1))) (n* z$3$1 w$23$1)))) (and (EQ z$3$1 null) (forall ((w$24$1 V)) (not (and (n* q w$24$1) (not (EQ q w$24$1))))))))))))) (and (n* a c) (n* b c) (ite (EQ u null) (EQ p a) (and (n* a u) (or (and (n* u p) (not (EQ u p)) (forall ((w$25$1 V)) (=> (and (n* u w$25$1) (not (EQ u w$25$1))) (n* p w$25$1)))) (and (EQ p null) (forall ((w$26$1 V)) (not (and (n* u w$26$1) (not (EQ u w$26$1))))))))) (ite (EQ v null) (EQ q b) (and (n* b v) (or (and (n* v q) (not (EQ v q)) (forall ((w$27$1 V)) (=> (and (n* v w$27$1) (not (EQ v w$27$1))) (n* q w$27$1)))) (and (EQ q null) (forall ((w$28$1 V)) (not (and (n* v w$28$1) (not (EQ v w$28$1)))))))))))) (EQ u v)))))
(check-sat)

(exit)
