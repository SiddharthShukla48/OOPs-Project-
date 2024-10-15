X = "India.is.my.country"
separated_values = X.split('.') 
comma_separated_values = ', '.join(separated_values)  
print("Comma-separated values:", comma_separated_values)


Y = "M.A.N.I.P.A.L"
character_to_remove = 'A'  
modified_string = Y.replace(character_to_remove, '') 
print("String after removing character:", modified_string)
