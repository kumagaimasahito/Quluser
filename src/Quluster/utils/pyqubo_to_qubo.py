import re

def pyqubo_to_qubo(pyqubo):
    regex = re.compile('\d+')
    trans = lambda coup: tuple(map(int, regex.findall(coup)))
    qubo = {
        (trans(coup_i), trans(coup_j)): coeff 
        for (coup_i, coup_j), coeff in pyqubo.items()
    }
    return qubo