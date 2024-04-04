from inverse_tracer import entry_point

# Ultimate Goal: Develop a custom solution that dynamically logs subsequent method calls that are **direct continuations of an initial method's execution path**.

# Recursive Definition - Methods that are **direct continuations of an initial method's execution path** are methods that satisfy one or more of the below properties:
    # 0. Base case: The initial method wrapped by entry_point automatically satisfied the definition of: **direct continuations of an initial method's execution path**
    # 1. Methods called within another method that recursively satisfies: **direct continuations of an initial method's execution path**
    # 2. Methods called as a subsequent chain (i.e. ().() ) of a method that recursively satisfies: **direct continuations of an initial method's execution path**
    # 3. Methods that are called as part of the process that delivers the direct response from an external endpoint that corresponds to a specific client-side method that makes a request to an external endpoint, given that the client-side method that makes the request satisfies: **direct continuations of an initial method's execution path**

def test_function():
    print()

class MyClass:
    def my_method(self):
        print("MyMethod called")
        return AnotherClass()

class AnotherClass:
    def another_method(self):
        print("AnotherMethod called")

@entry_point
def my_function():
    obj = MyClass()
    obj.my_method().another_method()  # This will be dynamically wrapped and logged

my_function()