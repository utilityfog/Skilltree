import asyncio
import traceback

from inverse_tracer import entry_point
from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def drive(self):
        pass

class Car(Vehicle):
    async def drive(self):
        await dummy_function()
        return f"The car {self.name} is driving on the road."

class Boat(Vehicle):
    def drive(self):
        return f"The boat {self.name} is sailing on the water."

# Ultimate Goal: Develop a custom solution that dynamically logs subsequent method calls that are **direct continuations of an initial method's execution path**.

# Recursive Definition - Methods that are **direct continuations of an initial method's execution path** are methods that satisfy one or more of the below properties:
    # 0. Base case: The initial method wrapped by entry_point automatically satisfied the definition of: **direct continuations of an initial method's execution path**
    # 1. Methods called within another method that recursively satisfies: **direct continuations of an initial method's execution path**
    # 2. Methods called as a subsequent chain (i.e. ().() ) of a method that recursively satisfies: **direct continuations of an initial method's execution path**
    # 3. Methods that are called as part of the process that delivers the direct response from an external endpoint that corresponds to a specific client-side method that makes a request to an external endpoint, given that the client-side method that makes the request satisfies: **direct continuations of an initial method's execution path**
    
async def main():
    await my_function()
    await dummy_function()
    
    # Create instances of Car and Boat
    car = Car("Tesla Model S")
    boat = Boat("Sunseeker Predator 108")
    
    # Demonstrate abstraction
    print(await car.drive())  # Output: The car Tesla Model S is driving on the road.
    print(boat.drive())  # Output: The boat Sunseeker Predator 108 is sailing on the water.

async def dummy_function(): # This method's name should not be logged
    print(f"Dummy Method called traceback: {traceback.print_stack()}")

async def test_function():
    print("Test Method called")

class MyClass:
    async def my_method(self):
        await test_function() # The name of this method should be logged as well.
        return AnotherClass()

class AnotherClass:
    async def another_method(self): # The name of this method should be logged as well.
        print("AnotherMethod called")
        
@entry_point
async def my_function():
    obj = MyClass()
    instance = await obj.my_method()
    await instance.another_method()  # This will be dynamically wrapped and logged
    
if __name__ == "__main__":
    asyncio.run(main())