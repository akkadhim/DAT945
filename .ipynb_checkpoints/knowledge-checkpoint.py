import os
import numpy as np
import pickle
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from time import time
from directories import dicrectories

class knowledge:
    def __init__(self, clause_weight_threshold, number_of_examples, accumulation,
                 clause_drop_p, clauses, T, s, epochs):
        self.clause_weight_threshold = clause_weight_threshold
        self.number_of_examples = number_of_examples
        self.accumulation = accumulation
        self.clause_drop_p = clause_drop_p
        self.clauses = clauses
        self.T = T
        self.s = s
        self.epochs = epochs

    def generate(self, X_train, i):
        target_word_clauses = []
        single_output_active = np.empty(1, dtype=np.uint32)
        single_output_active[0] = i
        # create TM-AE model
        tm = TMAutoEncoder(self.clauses, self.T, self.s, single_output_active, max_included_literals=3, accumulation=self.accumulation, feature_negation=False, platform='GPU', output_balancing=0.5)
                            
        training_time = 0
        for e in range(self.epochs):
            start_training = time()
            # train the TM-AE
            tm.fit(X_train, number_of_examples=self.number_of_examples)
            stop_training = time()
            training_time = training_time + (stop_training-start_training)

        # extract the clauses
        for j in range(self.clauses):
            weight = 0
            weight = tm.get_weight(0, j)
            related_literals = []
            for k in range(tm.clause_bank.number_of_literals):
                if tm.get_ta_action(j, k) == 1:
                    related_literals.append(k)
            target_word_clauses.append([weight, related_literals])

        # save result in pickle
        knowledge_filepath = os.path.join(dicrectories.knowledge , str(i) + '.pkl')
        with open(knowledge_filepath , "wb") as phase1file:
            pickle.dump(target_word_clauses, phase1file)
        return training_time, target_word_clauses