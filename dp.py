#!/usr/bin/env python3
import numpy as np


# Main function to simplify Pairwise Alignment Code
def main():

    # Values to send to the Class
    delta = 3
    pair_score = 10
    non_pair_score = 7
    sigma = np.asarray([[pair_score, non_pair_score, non_pair_score, non_pair_score],
                        [non_pair_score, pair_score, non_pair_score, non_pair_score],
                        [non_pair_score, non_pair_score, pair_score, non_pair_score],
                        [non_pair_score, non_pair_score, non_pair_score, pair_score]])
    dna_sequence_1 = "CACGCG"
    dna_sequence_2 = "CACCG"
    
    # Instantiate the Class with the variables from above
    pairwise_alignment = PairwiseAlignment(sigma, delta, dna_sequence_1, dna_sequence_2)
    
    # Call the pairwise_alignment function to align the DNA sequences
    optimal_alignment, optimal_score, optimal_actions = pairwise_alignment.pairwise_alignment()

    # Print the inputs and outputs
    print("\033[1mInputs\033[0m")
    print(f"DNA Sequence 1: {dna_sequence_1}")
    print(f"DNA Sequence 2: {dna_sequence_2}")
    print(f"Delta: {delta}")
    print(f"Sigma:")
    labels = ['A', 'C', 'G', 'T']
    print("    " + " ".join(f"\033[1m{label:3}\033[0m" for label in labels))
    for label, row in zip(labels, sigma):
        print(f"\033[1m{label}\033[0m " + " ".join(f"{val:3}" for val in row))
    print("\n")
    print(f"\033[1mPairwise Alignment\033[0m\n{optimal_alignment}\n")
    print(f"\033[1mPairwise Alignment Score\033[0m\n{optimal_score}\n")
    print(f"\033[1mActions for Pairwise Alignment\033[0m\n{', '.join(optimal_actions)}")


# Class Definition for the Pairwise Alignment functions
class PairwiseAlignment():
    
    
    # Init function that globalizes the delta, sigma_array, and dna_sequence variables
    def __init__(self, sigma_array: np.array, delta: int, dna_sequence_1: list, dna_sequence_2: list):
        
        self.delta = delta
        self.sigma_array = sigma_array
        self.dna_sequence_1 = []
        for term in dna_sequence_1:
            self.dna_sequence_1.append(term)
        self.dna_sequence_2 = []
        for term in dna_sequence_2:
            self.dna_sequence_2.append(term)


    def pairwise_alignment(self):

        # Instantiate the top and bottom alignment strings
        top_pairwise_alignment = ""
        bottom_pairwise_alignment = ""
        
        # Create copies of the DNA sequence arrays in order to modify them
        dna_sequence_1 = self.dna_sequence_1[:]
        dna_sequence_2 = self.dna_sequence_2[:]

        # Get the actions to produce the optimal alignment and the corresponding score
        score, actions = self.get_score_and_actions_from_dp(self.dna_sequence_1, self.dna_sequence_2)

        # Iterate for each action
        for action in actions:

            # Logic for a "deletion"
            if action == "deletion":

                # Add the current value from the first DNA sequence and a blank space to the running alignments
                top_pairwise_alignment = top_pairwise_alignment + dna_sequence_1.pop(0)
                bottom_pairwise_alignment = bottom_pairwise_alignment + "_"

            # Logic for an "insertion"
            elif action == "insertion":

                # Add the current value from the second DNA sequence and a blank space to the running alignments
                top_pairwise_alignment = top_pairwise_alignment + "_"
                bottom_pairwise_alignment = bottom_pairwise_alignment + dna_sequence_2.pop(0)

            # Logic for a "match"
            elif action == "match":

                # Add the current value from the first and second DNA sequences to the running alignments
                top_pairwise_alignment = top_pairwise_alignment + dna_sequence_1.pop(0)
                bottom_pairwise_alignment = bottom_pairwise_alignment + dna_sequence_2.pop(0)

        # Concentate the top and bottom alignments to get the final pairwise alignment
        pairwise_alignment = top_pairwise_alignment + "\n" + bottom_pairwise_alignment

        return pairwise_alignment, score, actions


    # Function that calculates the optimal actions and score of the optimal actions with Dynamic Programming
    def get_score_and_actions_from_dp(self, dna_sequence_1, dna_sequence_2):

        # Define variables for the current size of each DNA sequence
        dna_size_1 = len(dna_sequence_1)
        dna_size_2 = len(dna_sequence_2)

        # If either of the DNA sequences are empty, we have hit the base case
        if dna_size_1 == 0 or dna_size_2 == 0:
            
            # Initialize variables for the score and actions
            # This also hanldes the case where both DNA sequences have length 0
            base_case_score = 0
            actions = []

            # Logic for when there are remaining terms in the first DNA sequence
            if dna_size_1 != 0:

                # Calculate the score by multiplying the score of a deletion by the number of DNA terms remaining
                base_case_score = self.delta * dna_size_1

                # Generate the action array by adding a deletion action for each DNA term remaining 
                for _ in range(dna_size_1):
                    actions.append("deletion")

            # Logic for when there are remaining terms in the second DNA sequence
            elif dna_size_2 != 0:

                # Calculate the score by multiplying the score of a insertion by the number of DNA terms remaining
                base_case_score = self.delta * dna_size_2

                # Generate the action array by adding an insertion action for each DNA term remaining 
                for _ in range(dna_size_2):
                    actions.append("insertion")
            
            return base_case_score, actions
        
        # If it is not at the base case, do the following
        else:
            
            # Create DNA sequence arrays without the last term to do recursion
            recursive_dna_sequence_1 = dna_sequence_1[:-1]
            recursive_dna_sequence_2 = dna_sequence_2[:-1]

            # Use recursion to calculate the maximum pairwise alignment for the DNA sequences produced by taking each action
            recursive_deletion, deletion_prev_actions = self.get_score_and_actions_from_dp(recursive_dna_sequence_1, dna_sequence_2)
            recursive_insertion, insertion_prev_actions = self.get_score_and_actions_from_dp(dna_sequence_1 , recursive_dna_sequence_2)
            recursive_match, match_prev_actions = self.get_score_and_actions_from_dp(recursive_dna_sequence_1, recursive_dna_sequence_2)
            
            # Calculate the values of the deletion, insertion, and match actions by adding the delta or sigma value
            deletion = recursive_deletion + self.delta
            insertion = recursive_insertion + self.delta
            match = recursive_match + self.compute_sigma(dna_sequence_1[-1], dna_sequence_2[-1])
            
            # Find the maximum action
            max_value = max(deletion, insertion, match)
            
            # Define a set of the actions with their corresponding values and previous action lists for the current recursion
            action_values = {"deletion": [deletion, deletion_prev_actions], "insertion": [insertion, insertion_prev_actions], "match": [match, match_prev_actions]}
            
            # Create an array of actions that maximize the value 
            max_actions = [action for action, value in action_values.items() if value[0] == max_value]

            # Get the maximum action; uses the last action in the array to prioritize match actions
            max_action = max_actions[-1]

            # Use the previous actions list from the current maximum action and add the maximum action to the running list
            max_prev_actions = action_values[max_action][1]
            max_prev_actions.append(max_action)

            return max_value, max_prev_actions

    
    # Compute the sigma value by stripping the corresponding value from the sigma array
    def compute_sigma(self, l1, l2):
        
        # Define a set to use for indexing the sigma array 
        indexing = {"A":0, "C":1, "G":2, "T":3}
        
        # Find the indices of the letters from the indexing set
        index_l1 = indexing[l1]
        index_l2 = indexing[l2]
        
        # Obtain sigma using the indices from above to find the corresponding value in the sigma array
        sigma = self.sigma_array[index_l1][index_l2]
        
        return sigma


if __name__ == '__main__':
    main()