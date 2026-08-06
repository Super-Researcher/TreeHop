from enum import Enum


class NodeType(Enum):
    query = -1
    irrelevant_doc = 0
    relevant_doc = 1
    leaf = 2