# Project 3 README

Student 1: Maanit Shah (920584512)
Student 2: Sai Mannava (920691968)

## Project Status
ADDER15, ALU15, REGS15, IMM15, DECODER15, FLAGS8, CPU15 all done and working.

## Known Issues

## Circuit #1 (ADDER15):

## Circuit #2 (ALU15):
Implemented AND, OR, XOR, ADD, SUBTRACT, SLL, SRL, SRA.

## Circuit #3 (REGS15):

## Circuit #4 (IMM15):
Implemented imm2:0, imm3:0, imm6:0, imm9:0, and imm10:4. This could be used for the functions.

## Circuit #5 (DECODER15):

## Circuit #6 (FLAGS8):

## Circuit #7 (CPU15):
added every function onto the CPU and tried running it. Not if this works perfectly yet but I did as much as i was able to.

## References
- Youtube: https://www.youtube.com/watch?v=lvYCchzQTyE&ab_channel=DrCraigA.Evans

## Generative AI Use:

### Prompt #1 (DECODER15): Explain ROM word for various opcodes, and how opcodes are converted to get hex output bits.

### Response #1: 
A ROM word is the fixed binary pattern stored in ROM for each instruction opcode. Each opcode (like ADD, SUB, LOAD) is a binary code that serves as an address to the ROM. When the CPU sends the opcode to ROM, the ROM outputs the corresponding control word.

These control words are stored in binary form, but they are usually expressed in hexadecimal for readability. For example, if the opcode 0001 points to a control word 10101100 in ROM, this binary is represented as AC in hex.

Thus, the process is: Opcode (binary) → ROM address lookup → Control word (binary) → Hex output bits.

### Changes #1: We were able to determine hex code that needs to be put in the ROM for various instruction type like R-type, I-type, B-type.

### Prompt #2:
I am having issues with SLLI, can you check if my wiring is correct and splitter bits are divided correctly

### Response #2:
Stage-2 (shift-by-2) must use the output of Stage-1 as its source, not A again.
In your picture the two small “A” taps on the right suggest you’re building the shift-by-2 path from A; that bypasses Stage-1.

Here’s exactly how to wire the first two stages:
Stage S1 (shift by 1 when B0=1)
MUX S1 (15-bit)
Select: B0
In0: A[14:0] (no shift)
In1 (ShiftL1(A)): wire [14]←A13, [13]←A12, …, [1]←A0, [0]←0
Output: call this bus S1_OUT[14:0]
Stage S2 (shift by 2 when B1=1)
MUX S2 (15-bit)
Select: B1
In0: S1_OUT[14:0] ← (this is the chain!)
In1 (ShiftL2(S1_OUT)): build from S1_OUT, not A:
[14]←S1_OUT12, [13]←S1_OUT11, …, [2]←S1_OUT0, [1]←0, [0]←0
How to do the wiring neatly in Logisim
Put a splitter on A for S1’s In1 wiring.
Put a splitter on S1_OUT (the S1 MUX output) for S2’s In1 wiring.
Use a joiner (splitter flipped) to re-assemble each shifted bus.
Only one 0 goes into S1’s LSB; two 0s go into S2’s two LSBs.
Quick self-check
Set A = 0000 0000 0000 001.
B0=1,B1=0 → output 0000 0000 0000 010 (shift by 1).
B0=0,B1=1 → output 0000 0000 0000 100 (shift by 2).
B0=1,B1=1 → output 0000 0000 0001 000 (shift by 3, proving the chain).
Once S2 is correct, add S4 (B2) and S8 (B3) the same way—each time, take the previous stage’s output as the source, and zero-fill the bottom 2^k bits.

### Changes #2:
Changed the splitter bits and fixed stage s2 and s3 outputs.