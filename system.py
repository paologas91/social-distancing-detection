import os


def cls():
    """
    Clear the terminal output
    """
    os.system('cls' if os.name == 'nt' else 'clear')
