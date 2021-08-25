from vitrix.dynamics import affine_local_strain  # noqa
import numpy as np
from hyperjson import dumps

print(dumps([{"key": "value"}, 81, True]))

x = np.array([[1,2],[1,1]], dtype=np.float64)
y = np.array([[1,1],[0,1]], dtype=np.float64)

out = affine_local_strain(x, y)


print(out)