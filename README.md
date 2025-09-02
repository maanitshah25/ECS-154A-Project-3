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

## References
- Youtube: https://www.youtube.com/watch?v=lvYCchzQTyE&ab_channel=DrCraigA.Evans

## Generative AI Use:

### Prompt #1 (DECODER15): Explain ROM word for various opcodes, and how opcodes are converted to get hex output bits.

### Response #1: 
A ROM word is the fixed binary pattern stored in ROM for each instruction opcode. Each opcode (like ADD, SUB, LOAD) is a binary code that serves as an address to the ROM. When the CPU sends the opcode to ROM, the ROM outputs the corresponding control word.

These control words are stored in binary form, but they are usually expressed in hexadecimal for readability. For example, if the opcode 0001 points to a control word 10101100 in ROM, this binary is represented as AC in hex.

Thus, the process is: Opcode (binary) → ROM address lookup → Control word (binary) → Hex output bits.

### Changes #1: We were able to determine hex code that needs to be put in the ROM for various instruction type like R-type, I-type, B-type.
