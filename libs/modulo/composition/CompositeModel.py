import numpy as np

from .compose import o, x
from ..graph.network import get_ids, node_label, get_nodes


class CompositeModel(object):
    def __init__(self, diagram, methods, desc_ids=None, targ_ids=None):

        # Assign desc and targ ids
        if desc_ids is not None:
            self.desc_ids = desc_ids
        elif "desc_ids" in diagram.graph:
            self.desc_ids = diagram.graph["desc_ids"]
        else:
            self.desc_ids = get_ids(diagram, kind="desc")

        if targ_ids is not None:
            self.targ_ids = targ_ids
        elif "targ_ids" in diagram.graph:
            self.targ_ids = diagram.graph["targ_ids"]
        else:
            self.targ_ids = get_ids(diagram, kind="targ")

        self.desc_ids = list(self.desc_ids)
        self.targ_ids = list(self.targ_ids)

        self.feature_importances_ = self.extract_feature_importances(diagram)

        self.predict = _get_predict(methods, self.targ_ids)
        self.out_kind = self.extract_out_kind(diagram, self.targ_ids)

        print(self.out_kind)

        if self.out_kind in {"nominal", "mix"}:

            self.nominal_targ_ids = self.extract_nominal_targ_ids(
                diagram, self.targ_ids
            )

            self.predict_proba = _get_predict_proba(methods, self.nominal_targ_ids)
            self.classes_ = _get_classes(diagram, self.nominal_targ_ids)

        return

    def extract_feature_importances(self, diagram, aggregation=np.sum):
        fi = []
        for idx in self.desc_ids:
            fi_idx = [
                d.get("fi", 0)
                for src, tgt, d in diagram.edges(data=True)
                if d.get("idx", 0) == idx
            ]
            fi.append(aggregation(fi_idx))

        norm = np.linalg.norm(fi, 1)
        fi = fi / norm
        return fi

    @staticmethod
    def extract_nominal_targ_ids(diagram, targ_ids):
        nominal_targ_ids = [
            t
            for t in targ_ids
            if diagram.node[node_label(t, kind="data")]["type"] == "nominal"
        ]
        return nominal_targ_ids

    @staticmethod
    def extract_out_kind(diagram, targ_ids):
        out_kinds = [diagram.node[node_label(t, kind="data")]["type"] for t in targ_ids]
        out_kinds = set(out_kinds)

        if len(out_kinds) == 1:
            return out_kinds.pop()
        else:
            return "mix"


def _get_classes(diagram, nominal_targ_ids):

    nominal_targ_ids.sort()

    # The prob and vote nodes have complete information on the classes
    targ_classes_ = [
        diagram.node[node_label(t, kind="prob")]["classes"] for t in nominal_targ_ids
    ]

    if len(targ_classes_) == 1:
        return targ_classes_.pop()
    else:
        return targ_classes_


def _get_predict_proba(methods, nominal_targ_ids):

    nominal_targ_ids.sort()
    tgt_methods = [methods[node_label(t, kind="prob")] for t in nominal_targ_ids]

    if len(tgt_methods) == 1:
        # We can safely return the only method
        return tgt_methods.pop()
    else:
        # We need to package them together
        return x(*tgt_methods, return_type=list)


def _get_predict(methods, targ_ids):
    """
    Compose single predict function for a diagram.

    Parameters
    ----------
    diagram
    methods

    Returns
    -------

    """
    targ_ids.sort()

    tgt_methods = [methods[node_label(t, kind="data")] for t in targ_ids]

    return o(np.transpose, x(*tgt_methods, return_type=np.array))
