; ECS 154A subroutine.asm 
; Loads a sinwave into the LED matrix and repeatedly starts over
MAIN: ; Entry point to program
    ; Setup the stack at 32512 (I/O starts at 32736), not necessary 
    LI x2, 32512

    LI x3, STRING
    JAL x1, PUTSTR
    ; Main loop
MAINLOOP:
    ; Call GETCHAR
    JAL x1, GETCHAR
    ; Call PUTCHAR
    JAL x1, PUTCHAR
    ; Branch always to loop
    J MAINLOOP

; GETCHAR
; x1 - Return address
; x3 - Return character
GETCHAR:
    ; Push x7
    ADDI x2, x2, -1
    SW x7, 0(x2)
GETCHARLOOP:
    ; Loop until character
    ; Compare I flag
    SF x7, x0
    SRLI x7, x7, 6 
    ; Jump back to loop
    BEQ x7, x0, GETCHARLOOP
    ; Readchar
GETCHARREAD:
    ; Load keyboard address
    LI x7, 0x7FF0
    LW x3, 0(x7)
    ; Pop x7
    LW x7, 0(x2)
    ADDI x2, x2, 1
    ; Return
    JR x1

; PUTCHAR 
; x1 - Return address
; x3 - Holds char to put out
PUTCHAR:
    ; Push x7
    ADDI x2, x2, -1
    SW x7, 0(x2)
    ; Load screen address
    LI x7, 0x7FF1
    ; Write char to screen
    SW x3, 0(x7)
    ; Pop x2
    LW x7, 0(x2)
    ADDI x2, x2, 1
    ; Return 
    JR x1

; PUTSTR
; x1 - Return address
; x3 - Holds address of string
PUTSTR:
    ; Push x7
    ; Push x6
    ADDI x2, x2, -2
    SW x7, 1(x2)
    SW x6, 0(x2)
    ; Load screen address
    LI x7, 0x7FF1
PUTSTROUTCHAR:
    ; Load charater from string
    LW x6, 0(x3)
    ; Check  for null
    BEQ x0, x6, PUTSTRDONE
    ; Write to screen
    SW x6, 0(x7)
    ; Increment pointer
    ADDI x3, x3, 1
    ; Loop
    J PUTSTROUTCHAR
PUTSTRDONE:
    ; Pop x7
    ; Pop x6
    LW x7, 1(x2)
    LW x6, 0(x2)
    ADDI x2, x2, 2
    ; Return 
    JR x1

STRING:
    DAT 0x0048
    DAT 0x0065
    DAT 0x006C
    DAT 0x006C
    DAT 0x006F
    DAT 0x0020
    DAT 0x0057
    DAT 0x006F
    DAT 0x0072
    DAT 0x006C
    DAT 0x0064
    DAT 0x000A
    DAT 0x0000
