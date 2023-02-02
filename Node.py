class Node:

    def __init__(self, nodeid, nodedict, threshold_used=False):
        """
        :param nodeid: Node ID
        :param nodedict: Node attributes, dictionary
        :param threshold_used: boolean, used to select which attributes will be retained
        """

        # kind defined here
        # will be set later in set_kind()
        self.kind = None

        #
        self.nodeid = nodeid

        # select attributes
        if not threshold_used:
            attrs = ATTRS
        else:
            attrs = ATTRS_recalc

        # Add selected attributes as fields
        for a in attrs:
            if a in nodedict:  # Necessary because nodes contain either 'Holes' or 'Fractality' (exclusive)
                self.__dict__[a.lower()] = nodedict[a]

        # set yso and node level
        self.yso = "Class" in nodedict
        self.level = RESMAP[nodedict["Res"]]

        # define adjacent node dictionaries
        self.children = NodeDict()
        self.parents = NodeDict()

    def __repr__(self):
        """
        :return: n_<ID>
        """
        return "n_" + str(self.nodeid)

    def add_child_from_edges(self, child_key, link, nodelist):
        """
        Adds children to self.children based on edge data
        Called by DataLoader when use_threshold is False
        """
        child_node = nodelist[child_key]
        self.children[child_node] = {
            "dir": link["dir"],
            "weight": link["weight"],
            "level": link["level"]
        }
        child_node.parents[self] = self.children[child_key][1]

    def add_children(self, nodelist, threshold=0.75):
        """
        Adds children to self.children based on area overlap
        Verified to produce the same edges as the dataset when using the same threshold (0.75)
        (only when swapping polygons for nodes 437 and 624, suspected to have been a mistake)
        Called by DataLoader when use_threshold is True
        """
        if threshold == 0:
            # special case of strict comparison with 0 threshold (to avoid ending up with a single structure)
            comp = lambda x, y: x > y
        else:
            comp = lambda x, y: x >= y
        for n in nodelist:
            # area intersection as a proportion of smaller node
            intersection = round(self.polygon.intersection(n.polygon).area / n.polygon.area, 6)
            if comp(intersection, threshold) and self.level < n.level:
                link = {
                    "weight": intersection,
                    "level": n.level - self.level
                }
                self.children[n] = link
                n.parents[self] = link

    def apport_prive(self):
        """
        :return: Apport prive du noeud, en nombre de noeuds
        """

        struct_nodes = NodeDict()
        for n in self.connected():
            struct_nodes[n] = None

        # print(struct_nodes)

        for n in self.connected():
            if n.kind == 1 and n.nodeid != self.nodeid:
                src_node = NodeDict()
                src_node[n] = None
                struct_nodes -= n.children
                struct_nodes -= src_node
                # print(struct_nodes)
        return len(struct_nodes)

    def close_ancestors(self):
        """
        Add ancestors that aren't already parents to self.parents with the link attribute "dir" = 2
        """
        for p_i, p in self.parents:
            update = (p.close_ancestors() - self.parents)
            self.parents = self.parents.close(update)
        return self.parents.copy()

    def close_descendants(self):
        """
        Add descendants that aren't already children to self.children with the link attribute "dir" = 2
        """
        for c_i, c in self.children:
            update = (c.close_descendants() - self.children)
            self.children = self.children.close(update)
        return self.children.copy()

    def close_links(self):
        """
        Close links in both directions
        :return:
        """
        self.children = self.close_descendants()
        self.parents = self.close_ancestors()
        return self.children, self.parents

    def connected(self, conn=None):
        """
        return all connected nodes
        """
        if conn is None:
            conn = set()
        conn |= {self}

        for k, c in self.children:
            if c not in conn:
                conn |= c.connected(conn)

        for k, p in self.parents:
            if p not in conn:
                conn |= p.connected(conn)

        return conn

    def set_kind(self):
        """
        0: isolated
        1: source
        2: middle
        3: leaf
        """
        if (len(self.parents) == 0) and (len(self.children) == 0):
            self.kind = 0
        elif len(self.children) == 0:
            self.kind = 3
        elif len(self.parents) == 0:
            self.kind = 1
        else:
            self.kind = 2

    def stats(self):
        """
        Return various stats about this node, in a pandas Series
        Meant to be used inside DataFrame.groupby.apply
        :return: Series
        """

        _stats = pd.Series({

        })


class NodeDict:
    """
    Dictionary of related nodes (use this class for list of parents or children)
    Mostly allows easier, integer indexing
    """

    def __add__(self, other):
        """
        :param other: other nodedict
        :return: set union
        """
        union = set(self) | set(other)
        union_dict = NodeDict()
        for i, n in union:
            union_dict[n] = self[i][1]
        return union_dict

    def __contains__(self, item):
        """
        :param item: nodeid
        :return: whether it contains that node
        """
        return item in self.id_dict

    def __eq__(self, other):
        """
        :param other: the other NodeDict
        :return: compares the integer IDs of contents of NodeDict
        """
        return self.id_dict.keys() == other.id_dict.keys()

    def __getitem__(self, item):
        """
        implement subscript, use node_dict[node_id]
        """
        key = self.id_dict[item]
        return key, self.dict[key]

    def __init__(self):
        # dictionary of {node: link}
        self.dict = {}

        # dictionary of {id: node}, for much easier indexing
        self.id_dict = {}

    def __iter__(self):
        """
        :return: returns tuples of (node_id, node_object)
        """
        return iter(self.id_dict.items())

    def __len__(self):
        """
        :return: implements length
        """
        return len(self.dict)

    def __next__(self):
        """
        debug, forgot what for
        """
        print("next was called")

    def __repr__(self):
        """
        :return: list of node objects contained
        """
        return repr(list(self.dict.keys()))

    def __setitem__(self, key, value):
        """
        implement subscript assignment, use node_dict[node_object] = link_dictionary
        """
        self.id_dict[key.nodeid] = key
        self.dict[key] = value

    def __sub__(self, other):
        """
        :param other: other nodedict
        :return: set difference
        """
        diff = set(self) - set(other)
        diff_dict = NodeDict()
        for i, n in diff:
            diff_dict[n] = self[i][1]
        return diff_dict

    def close(self, other):
        """
        Add (other - self) to self with the attribute dir = 2
        :param other: other nodedict
        :return: closure with "dir": 2
        """
        sum_dict = NodeDict()
        sum_dict.id_dict = {**self.id_dict, **{k: v for k, v in other.id_dict.items() if k not in self.id_dict}}

        other_dict = {k: {**v} for k, v in other.dict.items() if k.nodeid not in self}
        for v in other_dict.values():
            v["dir"] = 2
        sum_dict.dict = {**other_dict, **self.dict}
        return sum_dict

    def copy(self):
        """
        :return: hopefully creates a copy of link attributes so they don't get modified
        """
        new_dict = {**self.dict}
        cpy = NodeDict()
        cpy.dict = new_dict
        cpy.id_dict = self.id_dict
        return cpy

    def keys(self):
        """
        :return: self.id_dict, as it contains what is essentially the keys
        """
        return self.id_dict

    def linear(self):
        """
        :return: whether this branch is linear (if all branches are linear, the structure is linear)
        """
        levels = [c.level for c in self.id_dict.values()]
        return len(levels) == len(set(levels))

    def missing(self):
        """
        :return: new NodeDict with only missing links
        """
        missing_linked = NodeDict()
        for c, l in self.dict.items():
            if l["dir"] == 2:
                missing_linked.id_dict[c.nodeid] = c
                missing_linked.dict[c] = l
        return missing_linked

    def reduction(self):
        """
        :return: new NodeDict with only direct links
        """
        direct_linked = NodeDict()
        for c, l in self.dict.items():
            if l["dir"] == 1:
                direct_linked.id_dict[c.nodeid] = c
                direct_linked.dict[c] = l
        return direct_linked


# attributes when creating from data
ATTRS = ['Kind', 'Res', 'Fractality', 'Holes', '_A', '_X', '_Y', 'R', '_B', '_Theta', 'Polygon', ]

# attributes when creating from threshold
ATTRS_recalc = ['Res', '_A', '_X', '_Y', 'R', '_B', '_Theta', 'Polygon', ]

# map resolutions to levels
RESMAP = {36.3: 1, 24.9: 2, 18.2: 3, 13.5: 4, 8.4: 5, 2.0: 6}
