from dataloader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from Struct import Struct

pd.options.display.max_rows = None

def check(structs, l):
    for k, v in structs.items():
        if l[0] < k:
            v.plot_struct()
            plt.show()
            print(k, v.structure_type())
            seen = input()
            if int(seen) != v.structure_type():
                raise Exception
            else:
                l[0] = k
            plt.clf()
            plt.close()


last = [0]
# check(last)

print("go")
s1 = DataLoader(75).df
s2 = DataLoader().df

p1 = DataLoader.unpickle_data(75)
p2 = DataLoader.unpickle_data()

structures, sources = DataLoader.load_data(75).values()
struct_orig, src_orig = DataLoader.load_data().values()
print("loaded")

tolin = lambda x: x.groupby(level=0).apply(lambda x: Struct(x, x.index[0][0], None))

print(set(struct_orig["linear"].droplevel(0).index) - set(structures["linear"].droplevel(0).index))