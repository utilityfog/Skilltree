from negator import Negate
from analyzer import test_halt, test_loop

def main():
    """Demo of the Halting Problem"""
    print(Negate(Callable=test_loop, Input=True)) # Negate will halt
    
    print(Negate(Callable=test_halt, Input=True)) # Negate will loop
    
    ### Halting Problem!
    print(Negate(Callable=Negate, Input=Negate))

if __name__ == "__main__":
    main()