import networkx as nx
import numpy as np

from functools import partial, reduce

from ..graph.network import get_ids, node_label
from ..composition import o, x
from ..utils import debug_print

VERBOSITY = 0


def base_inference_algorithm(g, q_desc_ids=None):

    # Convert the graph to its functions
    sorted_list = list(nx.topological_sort(g))

    msg = """
    sorted_list:    {}
    """.format(
        sorted_list
    )
    debug_print(msg, level=1, V=VERBOSITY)
    functions = {}

    if q_desc_ids is None:
        # desc ids not provided => all attributes which are diagrammatically identified as descriptive, are assumed
        # to be given as inputs
        q_desc_ids = list(get_ids(g, kind="desc"))
        # q_desc_ids.sort()
        print(q_desc_ids)

    for node_name in sorted_list:
        node = g.nodes(data=True)[node_name]

        if node.get("kind", None) == "data":
            if len(nx.ancestors(g, node_name)) == 0:
                functions[node_name] = _select_numeric(q_desc_ids.index(node["idx"]))
            else:
                # Select the relevant output
                previous_node = [t[0] for t in g.in_edges(node_name)][0]
                previous_t_idx = g.nodes()[previous_node]["tgt"]
                relevant_idx = previous_t_idx.index(node["idx"])

                functions[node_name] = o(
                    _select_numeric(relevant_idx), functions[previous_node]
                )

        elif node.get("kind", None) == "imputation":
            functions[node_name] = node["function"]

        elif node.get("kind", None) == "model":
            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            inputs = {g.nodes()[n]["tgt"][0]: functions[n] for n in previous_nodes}
            inputs = [
                inputs[k] for k in sorted(inputs)
            ]  # We need to sort to get the inputs in the correct order.

            inputs = o(np.transpose, x(*inputs, return_type=np.array))
            f = node["function"]
            functions[node_name] = o(f, inputs)

        elif node.get("kind", None) == "prob":
            # Select the relevant output
            prob_idx = node["idx"]
            prob_classes = node["classes"]

            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            previous_classes = [g.edges[t]["classes"] for t in g.in_edges(node_name)]
            previous_t_idx = [g.nodes()[n]["tgt"] for n in previous_nodes]

            inputs = [
                (functions[n], t, c)
                for n, t, c in zip(previous_nodes, previous_t_idx, previous_classes)
            ]

            for idx, (f1, t, c) in enumerate(inputs):
                f2 = o(_select_nominal(t.index(prob_idx)), f1)

                if len(c) < len(prob_classes):
                    f2 = o(_pad_proba(c, prob_classes), f2)

                inputs[idx] = f2

            f = partial(np.sum, axis=0)
            functions[node_name] = o(f, x(*inputs, return_type=np.array))

        elif node.get("kind", None) == "vote":
            # Convert probabilistic votes to single prediction
            previous_node = [t[0] for t in g.in_edges(node_name)][0]
            f = partial(node["function"], classes=node["classes"])
            functions[node_name] = o(f, functions[previous_node])

        elif node.get("kind", None) == "merge":
            merge_idx = node["idx"]
            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            previous_t_idx = [g.nodes()[n]["tgt"] for n in previous_nodes]

            inputs = [(functions[n], t) for n, t in zip(previous_nodes, previous_t_idx)]

            inputs = [
                o(_select_numeric(t_idx.index(merge_idx)), f) for f, t_idx in inputs
            ]
            inputs = o(np.transpose, x(*inputs, return_type=np.array))

            f = partial(np.mean, axis=1)
            functions[node_name] = o(f, inputs)

    return functions


# Helpers
def _pad_proba(classes, all_classes):
    def pad(X):
        idx = _map_classes(classes, all_classes)
        R = np.zeros((X.shape[0], len(all_classes)))
        R[:, idx] = X
        return R

    return pad


def _map_classes(classes, all_classes):
    sorted_idx = np.argsort(all_classes)
    matches = np.searchsorted(all_classes[sorted_idx], classes)
    return sorted_idx[matches]


def _select_numeric(idx):
    def select(X):
        if len(X.shape) > 1:
            return X[:, idx]
        elif len(X.shape) == 1:
            assert idx == 0
            return X

    return select


def _select_nominal(idx):
    def select(X):
        if isinstance(X, list):
            return X[idx]
        elif isinstance(X, np.ndarray):
            assert idx == 0
            return X

    return select
