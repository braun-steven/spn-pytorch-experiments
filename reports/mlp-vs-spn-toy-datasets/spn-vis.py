from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.io.Graphics import plot_spn

from spn.structure.Base import Sum, Product

from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

## Setup (A)

# Group 1
g11 = Gaussian(mean=0.1, stdev=0.1, scope=0)
g12 = Gaussian(mean=0.1, stdev=0.1, scope=1)
g13 = Gaussian(mean=0.1, stdev=0.1, scope=2)

# Group 2
g21 = Gaussian(mean=0.1, stdev=0.1, scope=0)
g22 = Gaussian(mean=0.1, stdev=0.1, scope=1)
g23 = Gaussian(mean=0.1, stdev=0.1, scope=2)

# Product layer
p11 = Product([g11, g12])
p12 = Product([g12, g13])

p21 = Product([g21, g22])
p22 = Product([g22, g23])

# Sum Layer
s1 = Sum([0.5, 0.5], [p11, p21])
s2 = Sum([0.5, 0.5], [p12, p22])

# Root Node
root = Product([s1, s2])

assign_ids(root)
rebuild_scopes_bottom_up(root)

# Plot
plot_spn(root, "spn-A.png")


## Setup (B)

# Product layer
p11 = Product([g11, g12])
p12 = Product([g12, g13])
p13 = Product([g11, g13])


p21 = Product([g21, g22])
p22 = Product([g22, g23])
p23 = Product([g21, g23])

# Sum Layer
s1 = Sum([0.5, 0.5], [p11, p21])
s2 = Sum([0.5, 0.5], [p12, p22])
s3 = Sum([0.5, 0.5], [p13, p23])

# Root Node
root = Product([s1, s2, s3])

assign_ids(root)
rebuild_scopes_bottom_up(root)

# Plot
plot_spn(root, "spn-B.png")
