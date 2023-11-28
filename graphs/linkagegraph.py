import math
import numpy as np
import pandas as pd
import igraph as ig
import time
from multiprocessing import Pool
from functools import partial

class LinkageGraph:

    def __init__(self, df_edges: pd.DataFrame, prob_threshold: float = 0.95, min_order: int = 2) -> None:
        """Create a list of connected graphs (of at least a specified order) based on a dataframe of edges

        Args:
            df_edges (pd.DataFrame): Dataframe of edges
            prob_threshold (float, optional): Minimum "probability" of overall match for an edge to be included. Defaults to 0.95.
            min_order (int, optional): Minimum order of connected graphs to be returned. Defaults to 2.
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
        self.edges["weight"] = self.odds_to_distance(2 ** self.edges["match_weight"])
        self.edges["weight_given_name"] = self.odds_to_distance(self.edges["bf_given_name"])
        self.edges["weight_family_name"] = self.odds_to_distance(self.edges["bf_family_name"])
        self.edges["weight_dob_d"] = self.odds_to_distance(self.edges["bf_dob_d"])
        self.edges["weight_dob_m"] = self.odds_to_distance(self.edges["bf_dob_m"])
        self.edges["weight_dob_y"] = self.odds_to_distance(self.edges["bf_dob_y"])
        self.edges["weight_gender"] = self.odds_to_distance(self.edges["bf_gender"])

        g = ig.Graph.DataFrame(
            self.edges[[
                "unique_id_l", 
                "unique_id_r", 
                "weight", 
                "match_probability",
                "false_match",
                "weight_given_name",
                "weight_family_name",
                "weight_dob_d",
                "weight_dob_m",
                "weight_dob_y",
                "weight_gender"
                ]], 
            directed=False,
            use_vids=False,
        )

        g.vs['cluster'] = (np.asarray(g.vs['name']) / 1000).astype(int).tolist()
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
     

    def plot_subgraph(self, index:int, target_path:str = None) -> any:
        """Method to return pretty plot of subgraph

        Args:
            index (int): Subgraph index
            target (str): filepath

        Returns:
            ig.Graph: _description_
        """
        graph = self.connected_graphs[index]

        id_gen = ig.UniqueIdGenerator()
        color_indices = [id_gen.add(value) for value in graph.vs["cluster"]]
        vertex_palette = ig.ClusterColoringPalette(len(id_gen))
        color_list = [vertex_palette[index] for index in color_indices]
        graph.vs["color"] = color_list
        edge_widths = np.array(graph.es["match_probability"]) * 4 + 1
        graph.es["width"] = edge_widths.astype(int)

        edge_color_dict = ["#000", "#c00"]

        visual_style = {}
        visual_style["vertex_size"] = 10
        visual_style["edge_color"] = [edge_color_dict[int(binary)] for binary in graph.es["false_match"]]
        visual_style["bbox"] = (300, 300)
        visual_style["margin"] = 40

        if(target_path == None):
            return ig.plot(graph, **visual_style)
        else:
            ig.plot(graph, target = target_path, **visual_style)
            return "Image saved."

   
    def get_measures(self, weight_name:str = None) -> pd.DataFrame:
        """Method to return graph measures

        Args:
            weight_name (_type_, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """

        measures = []
        counter = 1
        count_total = len(self.connected_graphs)

        print(f"Working on connected subgraph {counter} of {count_total}")

        for sg in self.connected_graphs:
            if counter % 1000 == 0:
                print(f"Working on connected subgraph {counter} of {count_total}")

            counter += 1

            result = [len(sg.vs),
                    len(sg.es),
                    sg.diameter(weights = None), 
                    sg.diameter(weights = "weight"),
                    sg.diameter(weights = "weight_given_name"),
                    sg.diameter(weights = "weight_family_name"),
                    sg.diameter(weights = "weight_dob_d"),
                    sg.diameter(weights = "weight_dob_m"),
                    sg.diameter(weights = "weight_dob_y"),
                    sg.diameter(weights = "weight_gender"),
                    sg.transitivity_undirected(),
                    sg.transitivity_avglocal_undirected(),
                    sg.assortativity_degree(),
                    sg.density(),
                    sum(sg.es.get_attribute_values('false_match')),
                    len(np.unique(np.asarray(sg.vs['cluster']))),
                    ]
            measures.append(result)

        df = pd.DataFrame.from_records(measures, columns=[
            "vertices", 
            "edges",
            "diameter", 
            "diameter_weight_total", 
            "diameter_weight_given_name", 
            "diameter_weight_family_name", 
            "diameter_weight_dob_d", 
            "diameter_weight_dob_m", 
            "diameter_weight_dob_y", 
            "diameter_weight_gender", 
            "transitivity", 
            "tri_cluster_coef", 
            "assortativity_degree", 
            "density", 
            "false_link_count",
            "distinct_entities",
        ])

        df["any_false_links"] = (df["false_link_count"] > 0)

        return df
    