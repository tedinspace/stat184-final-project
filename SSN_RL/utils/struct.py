def list2map(L):
    '''list to map; list item need field of "name" '''
    M={}
    for item in L:
        M[item.name]=item
    return M


def getNames(L):
    '''get names from a list'''
    names = set()
    for item in L:
        names.add(item.name)
    return list(names)