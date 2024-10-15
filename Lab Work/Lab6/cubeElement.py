
def cube_elements(*args):
    cubes = [x ** 3 for x in args] 
    return cubes

result = cube_elements(1, 2, 3, 4, 5)
print("Cubes of the elements:", result)
