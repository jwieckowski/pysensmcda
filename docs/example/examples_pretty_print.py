from sympy import Matrix
from IPython.display import display, Markdown

def pretty_print_alternative(alternative_results):
    print(f'Alternative index: {alternative_results[0]}')
    print(f'Criteria index: {alternative_results[1]}')
    print(f'Change: {alternative_results[2]}')
    print(f'Resulting decision matrix:') 
    display(Matrix(alternative_results[3]))

def pretty_print_alternative_removal(alternative_results):
    print(f'Alternative index: {alternative_results[0]}')
    print(f'Resulting decision matrix:') 
    display(Matrix(alternative_results[1]))

def pretty_print_weights(weights):
    print('Resulting weights vector (example):')
    display(Matrix(weights).T)

def pretty_print_weights_generation(initial_w, result_w):
    print('Initial weights vector:')
    display(Matrix(initial_w).T)
    print(f'Modified weight index: {result_w[0]}')
    print(f'Modification: {result_w[1]}')
    print('Resulting weights vector:')
    display(Matrix(result_w[2]).T)

def pretty_print_crit_removal(result):
    print(f'Removed criterion index: {result[0]}')
    print(f'Resulting decision matrix:') 
    display(Matrix(result[1]))
    print('Resulting weights vector:')
    display(Matrix(result[2]).T)

def pretty_print_crit_identification(result):
    print(f'Removed criterion index(es): {result[0]}')
    print(f'Correlation value: {result[1]}')
    print(f'Distance value: {result[2]}')
    print(f'Resulting decision matrix:') 
    display(Matrix(result[3]))
