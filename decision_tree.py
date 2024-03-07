from typing import Any, Dict, Optional, Self
import numpy as np


class RestaurantDomainEncoder:
    type_encoding = {"French": 0, "Thai": 1, "Burger": 2, "Italian": 3}
    est_encoding = {"0-10": 0, "10-30": 1, "30-60": 2, ">60": 3}
    price_encoding = {"$": 0, "$$": 1, "$$$": 2}
    pat_encoding = {"None": 0, "Some": 1, "Full": 2}
    yesno_encoding = {"No": 0, "Yes": 1}
    attribute_encoding = {"Alt": 0, "Bar": 1, "Fri": 2, "Hun": 3, "Pat": 4, "Price": 5, "Rain": 6, "Res": 7, "Type": 8, "Est": 9}
    attribute_decoding = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est"]
    def encode(self, xs: np.array) -> np.array:
        y = []
        for x in xs:
            y.append([self.yesno_encoding[x[0]],
                    self.yesno_encoding[x[1]], 
                    self.yesno_encoding[x[2]], 
                    self.yesno_encoding[x[3]], 
                    self.pat_encoding[x[4]], 
                    self.price_encoding[x[5]], 
                    self.yesno_encoding[x[6]], 
                    self.yesno_encoding[x[7]], 
                    self.type_encoding[x[8]],
                    self.est_encoding[x[9]],
                    self.yesno_encoding[x[10]]
                ])
        return np.array(y)


class DecisionTree:
    def __init__(self, attribute_to_test: int, children: Optional[Dict[Any, Self]]=None, leaf_value: Optional[Any]=None):
        self.children = children
        self.is_leaf = False
        if self.children is None:
            self.children = {}
            self.is_leaf = True
        self.attribute_to_test = attribute_to_test
        self.leaf_value = leaf_value


    def add_child(self, attribute_value: Any, child: Self):
        self.children[attribute_value] = child
        if self.is_leaf:
            self.is_leaf = False


    def __call__(self, x: np.array):
        if self.is_leaf:
            return self.leaf_value
        else:
            v = x[self.attribute_to_test]
            r = self.children[v]
            if isinstance(r, DecisionTree):
                return r(x)
            else:
                return r


class DecisionTreeLearning:
    def bool_entropy(self, p: float):
        assert p >= 0.0 and p <= 1.0
        if p > 0.99:
            return p * np.log2(p)
        if p < 0.01:
            return (1.0 - p) * np.log2(1.0 - p)
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


    def remainder(self, attr: int, examples: np.array):
        assert len(examples) > 0
        assert attr >= 0 and attr < len(examples[0])
        PN = {}
        M = len(examples)
        for row in examples:
            v = row[attr]
            if v not in PN:
                PN[v] = {"P": 0, "N": 0}
            label = row[-1]
            if label == RestaurantDomainEncoder.yesno_encoding["No"]:
                PN[v]["N"] += 1
            else:
                assert label == RestaurantDomainEncoder.yesno_encoding["Yes"]
                PN[v]["P"] += 1
        res = 0.0
        for attr_val in PN:
            A = PN[attr_val]["P"] + PN[attr_val]["N"]
            res += (float(A)/float(M)) * self.bool_entropy(float(PN[attr_val]["P"])/float(A))
        return res


    def info_gain (self, q: float, attr: int, examples: np.array):
        return self.bool_entropy(q) - self.remainder(attr, examples)


    def calc_q(self, examples) -> float:
        P_tot, N_tot = 0, 0
        for row in examples:
            label = row[-1]
            if label == RestaurantDomainEncoder.yesno_encoding["No"]:
                N_tot += 1
            else:
                assert label == RestaurantDomainEncoder.yesno_encoding["Yes"]
                P_tot += 1
        return float(P_tot) / float(N_tot + P_tot)


    def plurality_value(self, examples: np.array) -> Any:
        """Retuns the most frequent label in examples."""
        return np.bincount(examples[:, -1]).argmax()


    def do_all_examples_have_same_label(self, examples: np.array) -> bool:
        """True is all examples have the same label"""
        return len(set(examples[:, -1])) == 1


    def importance(self, attribute, examples: np.array) -> float:
        return self.info_gain(self.calc_q(examples), attribute, examples)


    def narrow_examples_by_attribute_value(self, examples: np.array, attribute: int, value: Any):
        mask = (examples[:, attribute] == value)
        return examples[mask, :] 


    def learn(self, examples: np.array, attributes: set, parent_examples: np.array) -> DecisionTree:
        if not isinstance(attributes, set):
            attributes = set(attributes)
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        if self.do_all_examples_have_same_label(examples):
            # Return the first example's classification since they're all the same.
            return examples[0][-1]
        if len(attributes) == 0:
            return self.plurality_value(parent_examples)
        attribute_scores = [self.importance(a, examples) for a in attributes]
        #print(f"attrs: {attributes}\tscores: {attribute_scores}")
        A = np.argmax(attribute_scores)
        tree = DecisionTree(A)
        for v in set(examples[:, A]):
            exs = self.narrow_examples_by_attribute_value(examples, A, v)
            subtree = DecisionTreeLearning().learn(exs, attributes - set([A]), examples)
            tree.add_child(v, subtree)
        return tree