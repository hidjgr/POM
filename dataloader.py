import pandas as pd
from Node import Node
from Struct import Struct
import pickle


class DataLoader:
    """
    If instantiated, loads data from base dataset pickles into a pandas dataframe indexed by structure and node ID.
    Structures are sorted in descending order by number of nodes, then assigned integer IDs. Within a structure, node
    IDs are sorted in descending order.

    Contains method to save such data as a pickle and a static method to load.
    """

    def __init__(self, threshold=None):
        """
        Create new DataLoader object from base dataset
        :param threshold    : Minimum overlap threshold (in % of smaller object) for inclusion (default 0.75, strict
                              comparison if = 0)
        """
        with open("data/dataset/Edges.pkl", "rb") as f:
            edges = pickle.load(f)

        with open("data/dataset/Nodes.pkl", "rb") as f:
            nodes = [Node(k, n) for k, n in pickle.load(f).items()]

        # must be saved for pickle filename
        self.threshold_used = threshold is not None
        self.threshold = threshold

        # no longer needed as it was intended

        # add edges from pickled edges or recompute overlap
        if not self.threshold_used:
            for k, e in edges.items():
                for kc, c in e.items():
                    nodes[k].add_child_from_edges(kc, c, nodes)
        else:
            # add children when areas overlap with sufficient threshold
            # fix for swapped polygons 624 and 437
            nodes[624].polygon, nodes[437].polygon = nodes[437].polygon, nodes[624].polygon
            for n in nodes:
                n.add_children(nodes, threshold/100)
            # end

            # # Annotate direct links after all children are added # #

            # This map serves to annotate direct links with 1 and indirect with 0. Integers are used to allow for a
            # third value (2) for missing transitive links
            # this_could_have_been_an_if_statement.py
            direct_map = {
                True: 1,
                False: 0
            }

            for n in nodes:
                if len(n.children) > 0:
                    # the above if statement avoids the error created by expanding an empty children list to two values
                    c_ids, children = map(set, tuple(zip(*iter(n.children))))
                    for c in children:
                        p_ids = set(tuple(zip(*iter(c.parents)))[0]) - {n.nodeid}
                        # c[id] is a tuple (node object, link dictionary), we set "dir" in that dictionary accordingly
                        n.children[c.nodeid][1]["dir"] = direct_map[p_ids.intersection(c_ids - {c.nodeid}) == set()]

                # Annotate node kinds (isolated, leaf, source) from parent and children lists
                n.set_kind()

        # create df from original Node object dictionary and reorder columns
        self.df = pd.DataFrame([{**vars(n), **{"node": n}} for n in nodes])
        # reorder and keep ID for index
        self.df = self.df[["nodeid"] + NODE_COLUMNS]
        self.index_structs()
        self.df["closed_links"] = self.df.node.map(lambda n: n.close_links())
        self.df["children"] = self.df.closed_links.map(lambda l: l[0])
        self.df["parents"] = self.df.closed_links.map(lambda l: l[1])
        self.df = self.df[NODE_COLUMNS]

    def index_structs(self):
        """
        Add index level for structures formed by connected nodes
        """

        # get connected nodes of each row (each node) into a frozenset to use as index
        groups = {r[1].node: frozenset(r[1].node.connected()) for r in self.df.iterrows()}

        # add them as a new column
        self.df["struct"] = self.df.node.map(lambda n: groups[n])

        # make multiindex with that column and node id
        self.df.set_index(["struct", "nodeid"], inplace=True)

        # add column of structure sizes using length of structures used in index, for each row (duplicates intended)
        self.df["structsize"] = [len(i[0]) for i in self.df.index]

        # sort by that column in descending order to sort structures by size, with merge sort (stable) to conserve order
        # of nodes
        self.df = self.df.sort_index(level=1, ascending=False) \
            .sort_values(["structsize"], kind="mergesort", ascending=False)

        # map structures to contiguous ids
        valueids = {k: s for k, s in zip(self.df.index.get_level_values(0).unique(), range(0, len(groups)))}

        # apply map to the index
        # the subscripts of i are the levels of the index
        self.df.index = self.df.index.map(lambda i: (valueids[i[0]], i[1]))

        # sort again and select extra columns out
        self.df = self.df.sort_index(ascending=[True, False])[NODE_COLUMNS]

    def pickle_data(self):
        """
        save dataframe to pickle
        normally used in pickle_dfs.py for all thresholds
        """
        filename = "data/saved/"

        if not self.threshold_used:
            print("saving original dataset")
            filename += "nodes_df_with_edges.pkl"
        else:
            filename += "threshold/" + str(int(self.threshold)) + ".pkl"
            print(f"saving dataset reconstructed with {str(self.threshold)} overlap threshold to {filename}")

        with open(filename, "wb") as f:
            pickle.dump(self.df, f)

    @staticmethod
    def load_data(threshold=None):
        """
        Loads useful views of data (structure types, sources...etc), calls unpickle_data()
        :param threshold:
        :return: data separated
        """
        df = DataLoader.unpickle_data(threshold)
        src_df = df[df.kind == 1].groupby(level=1).apply(lambda s: df.loc[
                   (s.index[0][0], [s.node.squeeze().nodeid] + list(s.children.squeeze().keys().keys())), :])\
            .rename_axis(index=["sourceid", "struct", "nodeid"])\
            .reorder_levels(["struct", "sourceid", "nodeid"])\
            .sort_index(ascending=[True, False, False])

        return {
            "structures": {
                "df": df,
                "isolated": DataLoader.select_by_type(df, 2, threshold),
                "linear": DataLoader.select_by_type(df, 1, threshold),
                "hierarchical": DataLoader.select_by_type(df, 0, threshold),
                "objects": DataLoader.to_structs(df, threshold)
            },
            "sources": {
                "df": src_df,
                "isolated": DataLoader.select_by_type(src_df, 2, threshold),
                "linear": DataLoader.select_by_type(src_df, 1, threshold),
                "hierarchical": DataLoader.select_by_type(src_df, 0, threshold),
                "objects": DataLoader.to_structs(src_df, threshold)
            },
        }

    @staticmethod
    def select_by_type(df, struct_type, threshold):
        """
        Given a multi-indexed DataFrame, returns structures of desired type
        :param df: DataFrame (structures, or sources)
        :param struct_type: 0: hierarchical, 1: linear, 2: isolated
        :param threshold: yeah
        :return: Filtered structure
        """
        if len(df.index[0]) == 2:
            selection = df.groupby(level=0).apply(
                lambda s: Struct(s.droplevel([0]), s.index[0][0], threshold).structure_type() == struct_type)
            selection = list(selection[selection].index)
            return df.loc[(selection, slice(None)), :]
        elif len(df.index[0]) == 3:
            selection = df.groupby(level=1).apply(
                lambda s: Struct(s.droplevel([0, 1]), s.index[0][0:2], threshold).structure_type() == struct_type)
            selection = list(selection[selection].index)
            return df.loc[(slice(None), selection, slice(None)), :]

    @staticmethod
    def to_structs(df, threshold=75):
        """
        Creates a (nested if needed) dictionary of Struct objects indexed by the corresponding struct and source id
        :param df: The DataFrame containing all your structures
        :param threshold: threshold
        :return: dictionary of Struct objects
        """
        if len(df.index[0]) == 2:
            return {st: Struct(df.xs(st), st, threshold) for st in df.index.get_level_values(0).unique()}
        elif len(df.index[0]) == 3:
            return {st: {src: Struct(df.xs(st).xs(src), (st, src), threshold) for src in df.xs(st).index
                    .get_level_values(0).unique()} for st in df.index.get_level_values(0).unique()}

    @staticmethod
    def unpickle_data(threshold=None):
        """
        Always use this if you're actually going to do something with the dataframe
        :param threshold: the threshold of the saved dataframe pickle
        :return: the desired dataframe
        """
        filename = "data/saved/"
        if threshold is None:
            filename += "nodes_df_with_edges.pkl"
        else:
            filename += "threshold/" + str(int(threshold)) + ".pkl"

        print(f"unpickling from {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)


NODE_COLUMNS = ['level', 'kind', 'yso', 'node', 'children', 'parents', 'fractality', 'holes', '_a', '_x', '_y', 'r',
                '_b', '_theta', 'polygon']
