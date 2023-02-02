import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob


class Struct:
    """
    Class representing a single structure
    Contains the structure as a dataframe slice and relevant methods
    """

    def __init__(self, struct, struct_id, threshold):
        """
        :param struct   : Structure, as a DataFrame slice
        :param struct_id: Used for naming exported files, passed separately for convenience
        :param threshold: Used for naming exported files, passed separately for convenience
        """
        self.threshold = threshold
        self.struct_id = struct_id
        self.struct = struct
        self.LVLMAP = {1: 36.3, 2: 24.9, 3: 18.2, 4: 13.5, 5: 8.4, 6: 2.0}
        self.r0 = 2

    def __repr__(self):
        """
        :return: __repr__() of the contained structure
        """
        return repr(self.struct)

    def __str__(self):
        """
        :return: __str__() of the contained structure
        """
        return str(self.struct)

    def dichotomie(self, inf, sup, func):
        """
        Used in ideal_fractality_by_scale()
        Performs binary search of the ideal fractality index for a given number of nodes by repeatedly setting the
        fractality index to the center of the domain and moving the appropriate bound to the current fractality index
        until convergence
        :param inf: lower bound
        :param sup: upper bound
        :param func: function used to compute number of nodes
        :return:
        """
        n = self.struct.shape[0]
        f = (inf + sup) / 2
        while sup - inf > 0.0000001:
            f = (inf + sup) / 2
            if func(f) > n:
                sup = f
            else:
                inf = f
        return f

    def ideal_fractality(self, func):
        """

        :param func: function used to compute number of nodes given a fractality index
        :return:
        """
        f = 1
        lim = 0
        if func(f) < self.struct.shape[0]:
            while func(f) < self.struct.shape[0]:
                f *= 2
            lim = f / 2
        elif func(f) > self.struct.shape[0]:
            while func(f) > self.struct.shape[0]:
                f /= 2
            lim = f * 2
        return self.dichotomie(*(sorted([lim, f]) + [func]))

    def level_diffs(self, func):
        """
        :return:
        """
        level_diffs = []
        ideal_level = []
        for lvl in self.struct.level.unique():
            level_diffs.append(
                abs(self.ideal_fractality(func) ** (lvl - 1) - self.struct[self.struct.level == lvl].shape[0]))
            ideal_level.append(self.ideal_fractality(func) ** (lvl - 1))
        return level_diffs, ideal_level

    def mesure1(self):
        """
        Fractality by levels
        :return:
        """
        level_diffs, ideal_level = self.level_diffs(self.noeuds_a0)
        return sum(level_diffs) / (sum(ideal_level) * 2)

    def mesure2(self):
        """
        Fractality by nodes
        :return:
        """
        # Alternate function, basic
        func = lambda f: f ** 5 + f ** 4 + f ** 3 + f ** 2 + f + 1
        a_0 = self.ideal_fractality(self.noeuds_a0)
        fract_total = []
        for u in self.struct[self.struct.level < 6].node:
            frac = 0
            for i_v, v in u.children.reduction():
                frac += 1 / a_0 ** (math.log(u.res / (v.res * self.r0), self.r0))
            fract_total.append(abs(frac - a_0))

        norm = self.struct[self.struct.level < 6].shape[0] * a_0
        return sum(fract_total) / norm

    def noeuds_a0(self, f):
        """
        Number of nodes in a perfect structure with fractality f, used in binary search
        :param f:
        :return: number of nodes for f
        """
        srclvl = self.struct.level.min()
        # maxlvl = self.struct.level.max()
        s0 = self.LVLMAP[srclvl]
        s = []
        exposants = [0] + [math.log(s0 / self.LVLMAP[i + srclvl], self.r0) for i in range(1, 6 - srclvl + 1)]
        f_list = [f] * len(exposants)
        return sum([f ** e for f, e in zip(f_list, exposants)])

    def plot_graph(self):
        """
        create plot of structure as graph
        call inside plot_struct()
        """

        plt.subplot(1, 2, 1)

        nodequeue = sorted(
            list(zip(self.struct[(self.struct.kind == 1) | (self.struct.kind == 0)].index,
                     self.struct[(self.struct.kind == 1) | (self.struct.kind == 0)].level)),
            key=lambda x: x[0], reverse=True)
        nodelist = [*nodequeue]

        while len(nodequeue) > 0:
            n = nodequeue[0]
            nodequeue.remove(n)
            n = n[0]
            children = sorted([(n.nodeid, n.level) for n in self.struct.loc[n].children.reduction().dict.keys()],
                              reverse=True, key=lambda x: x[0])
            nodequeue += children
            nodelist += children

        nodelist = sorted(nodelist, key=lambda x: x[1])

        rset = set(n for n in nodelist if nodelist.count(n) > 1)

        for r in rset:
            for c in range(nodelist.count(r) - 1):
                nodelist.pop(nodelist.index(r, nodelist.index(r) + 1))

        nodes = [i * 2 for i in range(len(nodelist))]
        nodelabels = {k: v for k, v in zip(nodes, [n[0] for n in reversed(nodelist)])}
        reverselabels = {v: k for k, v in nodelabels.items()}

        node_colors = [self.struct.yso.map(lambda y: {True: "orange", False: "royalblue"}[y])[nodelabels[n]] for n in
                       nodes]
        edgs = sorted(
            [(reverselabels[n.nodeid], reverselabels[c.nodeid]) for n in self.struct.node for i, c in n.children if
             n.children[i][1]["dir"] != 0])
        edge_colors = [{1: "k", 2: "r"}[self.struct.node[nodelabels[n_i]].children[nodelabels[c_i]][1]["dir"]] for
                       n_i, c_i in edgs]

        G = nx.DiGraph()

        last_level = 7
        invisible_nodes = []
        for n in nodes:
            n_id = nodelabels[n]
            for i in range(last_level - self.struct.level[n_id] - 1):
                inv_node = max(list(G.nodes) + [0]) + 1
                if inv_node % 2 == 0:
                    inv_node += 1
                empty_level = last_level - i - 1
                G.add_node(inv_node, layer=empty_level)
                invisible_nodes.append(inv_node)
            G.add_node(n, layer=self.struct.level[n_id])
            last_level = self.struct.level[n_id]

        for i in range(last_level - 1, 0, -1):
            inv_node = max(list(G.nodes) + [-1]) + 1
            G.add_node(inv_node, layer=i)
            invisible_nodes.append(inv_node)

        G.add_edges_from(edgs)
        pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors)
        nx.draw_networkx_nodes(G, pos, nodelist=invisible_nodes, alpha=0)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
        nx.draw_networkx_labels(G, pos, labels=nodelabels)
        plt.axis("equal")
        return G

    def plot_sky(self):
        """
        create plot of structure in space
        call inside plot_struct()
        """

        plt.subplot(1, 2, 2)
        plt.gcf().frameon = False
        for n in self.struct.node:
            plt.plot(*n.polygon.exterior.xy, "k")
        plt.axis("equal")
        plt.axis("off")

    def plot_struct(self):
        """
        Create graph plot and ellipse plot side by side
        """
        plt.clf()
        plt.figure(self.struct_id[0] if isinstance(self.struct_id, tuple) else self.struct_id)
        plt.gcf().set_size_inches(20, 10)
        self.plot_graph()
        self.plot_sky()

    def save_struct(self):
        """
        Save plot to appropriate file (depending on threshold and IDs)
        ./images/<threshold>/<structure ID>/struct.png if structure
        ./images/<threshold>/<structure ID>/<source ID>.png if source
        """

        import os

        structid = str(self.struct_id[0] if isinstance(self.struct_id, tuple) else self.struct_id)
        typename = {0: "hierarchical", 1: "linear", 2: "isolated"}[self.structure_type()]

        structpath = glob.glob("images/" + str(self.threshold) + "/" + structid + "_*")[0]
        srcname = str(self.struct_id[1]) if isinstance(self.struct_id, tuple) else "struct_" + structid
        filename = srcname + "_" + typename

        self.plot_struct()
        print("Saving struct",
              ":".join(map(str, self.struct_id)) if isinstance(self.struct_id, tuple) else self.struct_id,
              ", threshold", str(self.threshold) + " to " +
              structpath + "/" + filename + ".pdf")  # file name

        plt.savefig(structpath + "/" + filename + ".pdf")

    def stats(self):
        """
        Return various stats about this structure, in a pandas Series
        Meant to be used inside DataFrame.groupby.apply
        :return: Series
        """
        # TODO: distribution des fractalites
        # TODO: densité

        _stats = pd.Series({
            "fractalite": self.ideal_fractality(self.noeuds_a0),
            "mesure1": self.mesure1(),
            "mesure2": self.mesure2(),
            "mesure_min": min(self.mesure1(), self.mesure2()),
            "mesure_sum": self.mesure1() + self.mesure2(),
            "etoiles_prop": self.struct.yso[self.struct.yso].shape[0] / self.struct.kind[self.struct.kind == 3].shape[
                0],
            "etoiles_et_puits_prop":
                self.struct.level[(self.struct.kind == 3) & (self.struct.level > 4)].shape[0] /
                self.struct.kind[self.struct.kind == 3].shape[0],
            "etoiles_nbr": self.struct.yso[self.struct.yso].shape[0],
            "puits_nbr": self.struct.kind[self.struct.kind == 3].shape[0],
            "def_trans": sum(self.struct.children.map(lambda x: len(x.missing())))
        })

        if type(self.struct_id) == tuple:
            _stats["apport_prive"] = self.struct.loc[self.struct.index.max()].node.apport_prive() / self.struct.shape[0]

        return _stats

    def structure_type(self):
        """
        Return type of structure
        :return: 0: hierarchical, 1: linear, 2: isolated
        """

        if self.struct.shape[0] == 1:
            return 2
        else:
            chlist = []
            plist = []

            for c in self.struct.children:
                chlist += list(c.reduction())
            for p in self.struct.parents:
                plist += list(p.reduction())

            if len(chlist) > len(set(chlist)) or len(plist) > len(set(plist)):
                return 0
            else:
                return 1

    def yso_branch_prop(self):
        """
        :return: ratio of YSOs:branches
        """
        return self.struct[self.struct.yso].shape[0] / self.struct[self.struct.kind == 3].shape[0]
