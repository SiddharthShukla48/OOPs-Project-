class Node:
    def __init__(self, key):
        self.key = key  # Node's value.
        self.left = None  # Initialize the left child as None.
        self.right = None  # Initialize the right child as None.

class BinaryTree:
    def __init__(self):
        self.root = None  # Initialize the tree with no root.

    def insert(self, key):
        if self.root is None:  # If the tree is empty, the new node becomes the root.
            self.root = Node(key)
        else:
            self._insert(self.root, key)  # Recursively find the position to insert.

    def _insert(self, current, key):
        if key < current.key:  # If the key is smaller, go to the left subtree.
            if current.left is None:  # If there's no left child, insert here.
                current.left = Node(key)
            else:
                self._insert(current.left, key)  # Recursively insert in the left subtree.
        else:  # If the key is larger or equal, go to the right subtree.
            if current.right is None:  # If there's no right child, insert here.
                current.right = Node(key)
            else:
                self._insert(current.right, key)  # Recursively insert in the right subtree.

    def inorder_traversal(self):
        result = []  # List to store the traversal result.
        self._inorder(self.root, result)  # Perform the traversal starting from the root.
        return result

    def _inorder(self, node, result):
        if node:  # If the node exists, process it.
            self._inorder(node.left, result)  # Traverse the left subtree.
            result.append(node.key)  # Add the node's key to the result.
            self._inorder(node.right, result)  # Traverse the right subtree.

# Example usage
tree = BinaryTree()
tree.insert(50)
tree.insert(30)
tree.insert(70)
tree.insert(20)
tree.insert(40)
tree.insert(60)
tree.insert(80)
print(tree.inorder_traversal())  # Output: [20, 30, 40, 50, 60, 70, 80]
