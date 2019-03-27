def ziggy(s, r): 
    ''' Matthew's solution ''' 
    if r==1: 
        return s
    
    rows = [''] * r
    i = 0 

    for character in s: 
        rows[i] += character
        if i == 0: 
            row_change = 1
        elif i == r - 1: 
            row_chage = -1
        i += row_change

    return ''.join(rows)


print(ziggy("LAMBDASCHOOLISLIT", 4))
