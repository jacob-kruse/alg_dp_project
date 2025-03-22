#!/usr/bin/env python3
import numpy as np


# Main function to simplify Pairwise Alignment Code
def main():

    # Values to send to the Class
    alpha = 0   # 0/4
    beta = 4
    pair_score = 10
    non_pair_score = 2.5   # 2.5/0
    sigma = np.asarray([[pair_score, non_pair_score, non_pair_score, non_pair_score],
                        [non_pair_score, pair_score, non_pair_score, non_pair_score],
                        [non_pair_score, non_pair_score, pair_score, non_pair_score],
                        [non_pair_score, non_pair_score, non_pair_score, pair_score]])
    dna_sequence_1 = ["C", "A", "C", "G", "C", "G"]
    dna_sequence_2 = ["T", "C", "A", "C", "C", "G"]
    
    # Instantiate the Class with the variables from above
    pairwise_alignment = PairwiseAlignment(sigma, alpha, beta, dna_sequence_1, dna_sequence_2)
    
    # Call the pairwise_alignment function to align the DNA sequences
    optimal_alignment, optimal_score, optimal_actions = pairwise_alignment.pairwise_alignment()

    # Print the inputs and outputs
    print("\033[1mInputs\033[0m")
    print(f"DNA Sequence 1: {''.join(dna_sequence_1)}")
    print(f"DNA Sequence 2: {''.join(dna_sequence_2)}")
    print(f"Alpha: {alpha}")
    print(f"Delta: {beta}")
    print(f"Sigma:")
    for row in sigma:
        print(" ".join(f"{val:3}" for val in row))
    print("\n")
    print(f"\033[1mPairwise Alignment with Affine Gap Penalty\033[0m\n{optimal_alignment}\n")
    print(f"\033[1mPairwise Alignment Score with Affine Gap Penalty\033[0m\n{optimal_score}\n")
    print(f"\033[1mActions for Pairwise Alignment with Affine Gap Penalty\033[0m\n{', '.join(optimal_actions)}")


# Class Definition for the Pairwise Alignment functions
class PairwiseAlignment():
    
    
    # Init function that globalizes the sigma_array, alpha, beta, and dna_sequence variables
    def __init__(self, sigma_array: np.array, alpha: int, beta: int, dna_sequence_1: list, dna_sequence_2: list):
        
        self.alpha = alpha
        self.beta = beta
        self.sigma_array = sigma_array
        self.dna_sequence_1 = dna_sequence_1
        self.dna_sequence_2 = dna_sequence_2


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

        # Pass the DNA sequences to each action's Dynamic Programming function
        match, match_prev_actions = self.match_dp(dna_sequence_1, dna_sequence_2)
        insertion, insertion_prev_actions = self.insertion_dp(dna_sequence_1, dna_sequence_2)
        deletion, deletion_prev_actions = self.deletion_dp(dna_sequence_1, dna_sequence_2)

        # Find the maximum action
        max_value = max(deletion, insertion, match)
        
        # Define a set of the actions with their corresponding values and previous action lists for the current recursion
        action_values = {"deletion": [deletion, deletion_prev_actions], "insertion": [insertion, insertion_prev_actions], "match": [match, match_prev_actions]}
        
        # Create an array of actions that maximize the value 
        max_actions = [action for action, value in action_values.items() if value[0] == max_value]

        # Use the previous actions list from the maximum action
        max_prev_actions = action_values[max_actions[-1]][1]

        return max_value, max_prev_actions


    # Dynamic Programming function for the match action
    def match_dp(self, dna_sequence_1, dna_sequence_2):
            
        # Create DNA sequence arrays without the last term to do recursion
        recursive_dna_sequence_1 = dna_sequence_1[:-1]
        recursive_dna_sequence_2 = dna_sequence_2[:-1]

        # Define variables to hold the size of the recursed DNA sequences
        recursive_dna_size_1 = len(recursive_dna_sequence_1)
        recursive_dna_size_2 = len(recursive_dna_sequence_2)

        # Calcualte the sigma value for the last terms in the DNA sequences
        match_value = self.compute_sigma(dna_sequence_1[-1], dna_sequence_2[-1])

        # If both of the DNA sequences are empty, we have hit the base case
        if recursive_dna_size_1 == 0 and recursive_dna_size_2 == 0:

            return match_value, ["match"]
        
        # If it is not at the base case, do the following
        else:

            # Define an empty set to hold the possible actions with their corresponding values and action lists
            action_values = {}

            # Check to make sure a recursive deletion action is possible
            if recursive_dna_size_1 != 0:

                # Use recursion to calculate the maximum score of a previous deletion action
                recursive_deletion, deletion_prev_actions = self.deletion_dp(recursive_dna_sequence_1, recursive_dna_sequence_2)

                # Add the value of the current match score and insert the corresponding values to the action set
                deletion = recursive_deletion + match_value
                action_values["deletion"] = [deletion, deletion_prev_actions]
            
            # Check to make sure a recursive insertion action is possible
            if recursive_dna_size_2 != 0:

                # Use recursion to calculate the maximum score of a previous insertion action
                recursive_insertion, insertion_prev_actions = self.insertion_dp(recursive_dna_sequence_1, recursive_dna_sequence_2)

                # Add the value of the current match score and insert the corresponding values to the action set
                insertion = recursive_insertion + match_value
                action_values["insertion"] = [insertion, insertion_prev_actions]

            # Check to make sure a recursive match action is possible
            if recursive_dna_size_1 != 0 and recursive_dna_size_2 != 0:

                # Use recursion to calculate the maximum score of a previous match action
                recursive_match, match_prev_actions = self.match_dp(recursive_dna_sequence_1, recursive_dna_sequence_2)

                # Add the value of the current match score and insert the corresponding values to the action set
                match = recursive_match + match_value
                action_values["match"] = [match, match_prev_actions]   

            # Find the maximum value from the action set
            max_value = max(values[0] for values in action_values.values())
            
            # Create an array of actions that maximize the value 
            max_actions = [action for action, value in action_values.items() if value[0] == max_value]

            # Get the maximum action; uses the last action in the array to prioritize match actions
            max_action = max_actions[-1]

            # Use the previous actions list from the current maximum action and add a match action to the list
            max_prev_actions = action_values[max_action][1]
            max_prev_actions.append("match")

            return max_value, max_prev_actions


    # Dynamic Programming function for the insertion action
    def insertion_dp(self, dna_sequence_1, dna_sequence_2):

        # Create DNA sequence arrays without the last term to do recursion
        recursive_dna_sequence_1 = dna_sequence_1[:-1]
        recursive_dna_sequence_2 = dna_sequence_2[:-1]

        # Define variables to hold the size of the recursed DNA sequences
        recursive_dna_size_1 = len(recursive_dna_sequence_1)
        recursive_dna_size_2 = len(recursive_dna_sequence_2)

        # If both of the DNA sequences are empty, we have hit the base case
        if recursive_dna_size_1 == 0 and recursive_dna_size_2 == 0:

            return self.alpha, ["insertion"]
        
        # If it is not at the base case, do the following
        else:

            # Define an empty set to hold the possible actions with their corresponding values and action lists
            action_values = {}

            # Check to make sure a recursive deletion action is possible
            if recursive_dna_size_1 != 0:

                # Use recursion to calculate the maximum score of a previous deletion action
                recursive_deletion, deletion_prev_actions = self.deletion_dp(dna_sequence_1, recursive_dna_sequence_2)
                
                # Add the value of an insertion action and insert the corresponding values to the action set
                deletion = recursive_deletion + self.alpha
                action_values["deletion"] = [deletion, deletion_prev_actions]            
            
            # Check to make sure a recursive insertion action is possible
            if recursive_dna_size_2 != 0:

                # Use recursion to calculate the maximum score of a previous insertion action
                recursive_insertion, insertion_prev_actions = self.insertion_dp(dna_sequence_1, recursive_dna_sequence_2)
                
                # Add the value of an insertion action and insert the corresponding values to the action set
                insertion = recursive_insertion + self.beta
                action_values["insertion"] = [insertion, insertion_prev_actions]

            # Check to make sure a recursive match action is possible
            if recursive_dna_size_1 != 0 and recursive_dna_size_2 != 0:

                # Use recursion to calculate the maximum score of a previous match action
                recursive_match, match_prev_actions = self.match_dp(dna_sequence_1, recursive_dna_sequence_2)
                
                # Add the value of an insertion action and insert the corresponding values to the action set
                match = recursive_match + self.alpha
                action_values["match"] = [match, match_prev_actions]
            
            # Find the maximum value from the action set
            max_value = max(values[0] for values in action_values.values())

            # Create an array of actions that maximize the value 
            max_actions = [action for action, value in action_values.items() if value[0] == max_value]

            # Get the maximum action; uses the last action in the array to prioritize match actions
            max_action = max_actions[-1]

            # Use the previous actions list from the current maximum action and add an insertion action to the list
            max_prev_actions = action_values[max_action][1]
            max_prev_actions.append("insertion")

            return max_value, max_prev_actions


    # Dynamic Programming function for the deletion action
    def deletion_dp(self, dna_sequence_1, dna_sequence_2):

        # Create DNA sequence arrays without the last term to do recursion
        recursive_dna_sequence_1 = dna_sequence_1[:-1]
        recursive_dna_sequence_2 = dna_sequence_2[:-1]

        # Define variables to hold the size of the recursed DNA sequences
        recursive_dna_size_1 = len(recursive_dna_sequence_1)
        recursive_dna_size_2 = len(recursive_dna_sequence_2)

        # If both of the DNA sequences are empty, we have hit the base case
        if recursive_dna_size_1 == 0 and recursive_dna_size_2 == 0:

            return self.alpha, ["deletion"]
        
        # If it is not at the base case, do the following
        else:

            # Define an empty set to hold the possible actions with their corresponding values and action lists
            action_values = {}
            
            # Check to make sure a recursive deletion action is possible
            if recursive_dna_size_1 != 0:

                # Use recursion to calculate the maximum score of a previous deletion action
                recursive_deletion, deletion_prev_actions = self.deletion_dp(recursive_dna_sequence_1, dna_sequence_2)
                
                # Add the value of an deletion action and insert the corresponding values to the action set
                deletion = recursive_deletion + self.beta
                action_values["deletion"] = [deletion, deletion_prev_actions]
            
            # Check to make sure a recursive insertion action is possible
            if recursive_dna_size_2 != 0:

                # Use recursion to calculate the maximum score of a previous insertion action
                recursive_insertion, insertion_prev_actions = self.insertion_dp(recursive_dna_sequence_1, dna_sequence_2)
                
                # Add the value of an deletion action and insert the corresponding values to the action set
                insertion = recursive_insertion + self.alpha
                action_values["insertion"] = [insertion, insertion_prev_actions]

            # Check to make sure a recursive match action is possible
            if recursive_dna_size_1 != 0 and recursive_dna_size_2 != 0:

                # Use recursion to calculate the maximum score of a previous match action
                recursive_match, match_prev_actions = self.match_dp(recursive_dna_sequence_1, dna_sequence_2)
                
                # Add the value of an deletion action and insert the corresponding values to the action set
                match = recursive_match + self.alpha
                action_values["match"] = [match, match_prev_actions]
            
            # Find the maximum value dfrom the action set
            max_value = max(values[0] for values in action_values.values())
            
            # Create an array of actions that maximize the value 
            max_actions = [action for action, value in action_values.items() if value[0] == max_value]

            # Get the maximum action; uses the last action in the array to prioritize match actions
            max_action = max_actions[-1]

            # Use the previous actions list from the current maximum action and add a deletion action to the list
            max_prev_actions = action_values[max_action][1]
            max_prev_actions.append("deletion")

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