import sys
import numpy as np
if len(sys.argv)!=3:
    print("Give features #classes\n" ,file=sys.stderr)
    sys.exit(1)

features=np.array(sys.argv[1].split(',')).astype(int)
nclasses=int(sys.argv[2])

print(len(sys.argv))

print(sys.argv[0])
print(type(features))
print(features)
print(len(features))

print(type(nclasses))
print(int(nclasses))

display(type(sys.argv))
