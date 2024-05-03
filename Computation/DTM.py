import big_o

from Runtime import log_runtime, complexity_estimator

# class TwoTapeDTM:
#     def __init__(self, input_string):
#         self.tape1 = list(input_string)  # First tape with input string
#         self.tape2 = []  # Second tape, initially empty
#         self.head1 = 0  # Head position on tape 1
#         self.head2 = 0  # Head position on tape 2

    # def visualize(self):
    #     # Visualize the content of the tapes and the positions of the heads
    #     tape1_visual = ''.join(self.tape1)
    #     tape2_visual = ''.join(self.tape2 + ['_'] * (len(tape1_visual) - len(self.tape2)))
    #     print(f"Tape 1: {tape1_visual} | Head at: {self.head1}")
    #     print(f"Tape 2: {tape2_visual} | Head at: {self.head2}")
    #     print('-' * 40)
        
#     @log_runtime
#     def process(self):
#         self.visualize()
#         # Move through tape 1 and copy '0's to tape 2
#         while self.head1 < len(self.tape1) and self.tape1[self.head1] == '0':
#             self.tape2.append('0')  # Write '0' to tape 2
#             self.tape1[self.head1] = 'X'  # Mark '0' as processed on tape 1
#             self.head1 += 1
#             self.head2 += 1
#             self.visualize()

#         # Transition to processing '1's
#         while self.head1 < len(self.tape1) and self.tape1[self.head1] == '1':
#             if self.head2 > 0:
#                 self.tape2[self.head2 - 1] = 'X'  # Mark '0' as used on tape 2
#                 self.head2 -= 1  # Move tape 2 head left
#             else:
#                 print("Result: Reject (More '1's than '0's)")
#                 return False  # Reject due to imbalance
#             self.head1 += 1
#             self.visualize()

#         # Final decision based on remaining symbols in tapes
#         if self.head1 == len(self.tape1) and self.head2 == 0:
#             print("Result: Accept")
#             return True  # Accept the string
#         print("Result: Reject (Leftover '0's on tape 2 or extra characters)")
#         return False  # Reject due to leftover '0's or extra characters

# # Test the DTM with different strings
# print("Testing '000111'")
# dtm = TwoTapeDTM("000111")
# dtm.process()  # Expected: True

# print("\nTesting '00111'")
# dtm = TwoTapeDTM("00111")
# dtm.process()  # Expected: False

class TwoTapeDTM:
    def __init__(self, input_string):
        self.tape1 = list(input_string)  # First tape with input string
        self.tape2 = []  # Second tape, initially empty
        self.head1 = 0  # Head position on tape 1
        self.head2 = 0  # Head position on tape 2
        self.operation_count = 0  # Initialize operation count
        
    def visualize(self):
        # Visualize the content of the tapes and the positions of the heads
        tape1_visual = ''.join(self.tape1)
        tape2_visual = ''.join(self.tape2 + ['_'] * (len(tape1_visual) - len(self.tape2)))
        print(f"Tape 1: {tape1_visual} | Head at: {self.head1}")
        print(f"Tape 2: {tape2_visual} | Head at: {self.head2}")
        print('-' * 40)

    @log_runtime
    def process(self):
        self.visualize()
        while self.head1 < len(self.tape1) and self.tape1[self.head1] == '0':
            self.tape2.append('0')
            self.tape1[self.head1] = 'X'
            self.head1 += 1
            self.head2 += 1
            self.operation_count += 1  # Increment for each operation
            self.visualize()

        while self.head1 < len(self.tape1) and self.tape1[self.head1] == '1':
            if self.head2 > 0:
                self.tape2[self.head2 - 1] = 'X'
                self.head2 -= 1
                self.operation_count += 1  # Increment for each operation
            else:
                print("Result: Reject (More '1's than '0's)")
                return False
            self.head1 += 1
            self.visualize()

        if self.head1 == len(self.tape1) and self.head2 == 0:
            print("Result: Accept")
            return True
        print("Result: Reject (Leftover '0's on tape 2 or extra characters)")
        return False
    
class SingleTapeDTM:
    def __init__(self, input_string):
        self.tape = list(input_string)  # Tape with input string
        self.head = 0  # Head position on tape

    def visualize(self):
        # Visualize the content of the tape and the position of the head
        tape_visual = ''.join(self.tape)
        print(f"Tape: {tape_visual} | Head at: {self.head}")
        print('-' * 40)

    @log_runtime
    def process(self):
        try:
            while True:
                # Find the next '0' and mark it
                while self.head < len(self.tape) and self.tape[self.head] != '0':
                    self.head += 1
                if self.head >= len(self.tape):
                    break  # No more '0's to process
                self.tape[self.head] = 'X'
                # mark_zero_position = self.head
                self.visualize()

                # Find the corresponding '1'
                self.head += 1  # Move head to the right from the marked '0'
                while self.head < len(self.tape) and self.tape[self.head] != '1':
                    self.head += 1
                if self.head >= len(self.tape):
                    print("Result: Reject (Unmatched '0's)")
                    return False
                self.tape[self.head] = 'X'
                self.visualize()

                # Reset head to the start of the tape for next cycle
                self.head = 0

        except IndexError:
            print("Result: Reject (Tape out of bounds error)")
            return False

        # Final check for any remaining unmarked '1's
        if '1' in self.tape:
            print("Result: Reject (Unmatched '1's)")
            return False

        print("Result: Accept")
        return True

print("Testing '000111'")
dtm = TwoTapeDTM("000111")
dtm.process()  # Expected: True

print("\nTesting '00111'")
dtm = TwoTapeDTM("00111")
dtm.process()  # Expected: False

# STDTM
print("Testing '000111'")
stdtm = SingleTapeDTM("000111")
stdtm.process()  # Expected: True

print("\nTesting '00111'")
stdtm = SingleTapeDTM("00111")
stdtm.process()  # Expected: False