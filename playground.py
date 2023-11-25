def main(algorithm='gw', num_of_nodes=10, *args, **kwargs):
    print(f"Algorithm: {algorithm}")
    print(f"Number of Nodes: {num_of_nodes}")

    # Additional arguments
    for arg in args:
        print(f"Additional arg: {arg}")

    # Additional keyword arguments
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Example usage
main('brute_force', 15, 'extra_arg1', 'extra_arg2', param1='value1', param2='value2')
