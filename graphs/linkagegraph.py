import numpy as np
import pandas as pd
import igraph as ig

class LinkageGraph:

    def __init__(self, df_edges: pd.DataFrame, prob_threshold: float = 0.95, min_order: int = 1) -> None:
        """Create a list of connected graphs (of at least a specified order) based on a dataframe of edges

        Args:
            df_edges (pd.DataFrame): Dataframe of edges
            prob_threshold (float, optional): Minimum "probability" of overall match for an edge to be included. Defaults to 0.95.
            min_order (int, optional): Minimum order of connected graphs to be returned. Defaults to 1.
        """

        self.prob_threshold = prob_threshold
        self.min_order = min_order

        self.edges = df_edges.loc[(df_edges["match_probability"] >= self.prob_threshold)].copy()
        self.edges = self.edges.astype({
            'unique_id_l': 'int64',
            'unique_id_r': 'int64',
            'cluster_l': 'int64',
            'cluster_r': 'int64'
        })
        self.edges["false_match"] = (self.edges["cluster_l"] != self.edges["cluster_r"])
        self.edges["match_probability_rounded"] = np.round(self.edges["match_probability"], 2)
        self.edges["match_probability_disp"] = np.floor(self.edges["match_probability"] * 4) + 1
        self.edges["weight"] = self.odds_to_distance(2 ** self.edges["match_weight"])

        g = ig.Graph.DataFrame(
            self.edges[[
                "unique_id_l", 
                "unique_id_r", 
                "weight", 
                "match_probability_rounded", 
                "match_probability_disp", 
                "false_match"
                ]], 
            directed=False,
            use_vids=True,
        )

        g.vs['name'] = g.vs.indices
        self.connected_graphs = g.decompose(minelements = self.min_order)

    @staticmethod
    def odds_to_distance(prob_ratio: float) -> float:
        """Convert Bayes' factor to a distance score on [0.0001,1.0001], that is a
        high Bayes' factor would result in a value nearer 0 and low 
        Bayes' factor would result in a value nearer 1.

        Args:
            prob_ratio (float): Bayes factor (ratio of probabilities)

        Returns:
            float: Distance score on [0.0001,1.0001].
        """
        score = 1.0001 - prob_ratio / (1 + prob_ratio)
        return score

    @staticmethod
    def edge_betweenness_modularity(g: ig.Graph, weight_field: str) -> float:
        """Calculates the modularity of the clustering resulting from 
        partitioning the graph into two clusters by removing the edge(s) with 
        highest betweenness.

        Args:
            g (ig.Graph): A connected graph
            weight_field (str): The field name to use for weights, if any.

        Returns:
            float: The edge betweenness for the clustering into 2 clusters
        """
        
        if (isinstance(weight_field, str)):
            g.es.set_attribute_values("_weight_mod", [1.0002 - val for val in g.es.get_attribute_values(weight_field)])

            clustering = g.community_edge_betweenness(clusters = 2, weights = weight_field).as_clustering()
            modularity =  g.modularity(clustering, weights = "_weight_mod")

            del(g.es["_weight_mod"])
        else:
            clustering = g.community_edge_betweenness(clusters = 2).as_clustering()
            modularity =  g.modularity(clustering)

        return modularity

    @staticmethod
    def get_distinct_entities(vertices: list) -> int:
        """Returns the number of distinct entities in a graph.
        This is accomplished by recognising that the vertice

        Args:
            vertices (_type_): _description_

        Returns:
            _type_: _description_
        """
        return len(np.unique(np.floor(np.asarray(vertices) / 1000)))

    @staticmethod
    def is_super_bridge(g: ig.Graph, bridge: tuple[int, int], subgraph_min_order: int) -> bool:
        """Calcualtes if a specified bridge partitions a graph into two sub-graphs 
        each with order (number of vertices) at least a specified value, deemed a "super bridge".

        Args:
            g (ig.Graph): A connected graph
            bridge (tuple[int, int]): A bridge specified in terms of source and target vertices
            subgraph_min_order (int): The minimum number of vertices in each resulting sub-graph 
            for the bridge to be deemed a "super bridge".

        Returns:
            bool: True if the bridge is a "super bridge", otherwise False.
        """
        edge_id = g.get_eid(bridge[0], bridge[1])
        attr = g.es[edge_id].attributes()
        g.es[edge_id].delete()
        dg = g.decompose()
        g.add_edges([bridge])
        edge_id = g.get_eid(bridge[0], bridge[1])
        g.es[edge_id].update_attributes(attr)

        return ((len(dg[0].vs) >= subgraph_min_order) & (len(dg[1].vs) >= subgraph_min_order))

    @staticmethod
    def super_bridges(g: ig.Graph, subgraph_min_order: int = 3) -> int:
        """Counts the number of "super bridges" (see :is_super_bridge:) in a 
        connected graph.

        Args:
            g (ig.Graph): A connected graph
            subgraph_min_order (int, optional): The minimum number of vertices in each 
            resulting sub-graph for the bridge to be deemed a "super bridge". 
            Defaults to 3.

        Returns:
            int: The number of "super bridges" in the connected graph.
        """
        bridge_ids = g.bridges()
        bridge_vertices = []

        for edge_id in bridge_ids:
            bridge_vertices.append((g.es[edge_id].source, g.es[edge_id].target))

        n_super_bridges = 0

        for b in bridge_vertices:
            n_super_bridges += int(LinkageGraph.is_super_bridge(g, b, subgraph_min_order))
        
        return n_super_bridges

    def get_measures(self, weight_field_name:str = None, subgraph_min_order: int = 3) -> pd.DataFrame:
        """Method to return graph measures

        Args:
            weight_field_name (_type_, optional): _description_. Defaults to None.
            subgraph_min_order (int, optional): _description_. Defaults to 3.

        Returns:
            pd.DataFrame: _description_
        """

        sg_measure = []

        for sg in self.connected_graphs:
            measures = [len(sg.vs),
                        len(sg.es),
                        self.super_bridges(sg, subgraph_min_order), 
                        sg.diameter(weights = weight_field_name), 
                        sg.transitivity_undirected(),
                        sg.transitivity_avglocal_undirected(),
                        sg.assortativity_degree(),
                        sg.density(),
                        self.edge_betweenness_modularity(sg, weight_field_name),
                        sum(sg.es.get_attribute_values("false_match")),
                        self.get_distinct_entities(sg.vs['name']),
                        ]
            sg_measure.append(measures)

        df = pd.DataFrame(sg_measure, columns = [
            "vertices", 
            "edges", 
            "super_bridge_count", 
            "diameter", 
            "transitivity", 
            "tri_cluster_coef", 
            "assortativity_degree", 
            "density", 
            "cluster_edge_betweenness_modularity", 
            "false_match_count",
            "distinct_entities",
        ])

        df["any_false_matches"] = (df["false_match_count"] > 0)

        return df

    def get_graph(self, index:int) -> ig.Graph:
        return self.connected_graphs[index]

