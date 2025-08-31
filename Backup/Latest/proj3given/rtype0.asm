; ECS 154A rtype0.asm
; Just executes the r-type instructions on x0, esssentially NOP
MAIN: ; Entry point to program
    ADD x0, x0, x0
    SUB x0, x0, x0
    AND x0, x0, x0
    OR  x0, x0, x0
    XOR x0, x0, x0
    SLL x0, x0, x0
    SRL x0, x0, x0
    SRA x0, x0, x0