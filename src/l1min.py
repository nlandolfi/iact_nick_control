from l1 import l1
from cvxopt import matrix
from cvxopt import normal

m, n = 500, 200
P, q = normal(m,n), normal(m,1)
u = l1(P,q)
print u
