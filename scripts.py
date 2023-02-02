from dataloader import DataLoader


def pickle(threshold_range):
    for t in threshold_range:
        DataLoader(t).pickle_data()


def savefigs(threshold_range):
    for t in threshold_range:
        structures, sources = DataLoader.load_data(t).values()
        structs = DataLoader.to_structs(structures["df"], t)
        srcs = DataLoader.to_structs(sources["df"], t)

        for s in structs.values():
            s.save_struct()
        for s in srcs.values():
            for n in s.items():
                n[1].save_struct()


pickle([None])
pickle([75])
