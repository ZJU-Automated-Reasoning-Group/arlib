; cvc5 --produce-abducts
(set-logic UF)
(declare-const A Bool)
(declare-const B Bool)
(declare-const C Bool)
(assert (=> A C))
(get-abduct D (=> A B))