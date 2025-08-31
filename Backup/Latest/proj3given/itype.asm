; ECS 154A itype.asm 
; Just executes immediate instructions
MAIN: ; Entry point to program
    ADDI x1,x0,1
    ADDI x2,x0,-1
    ANDI x3,x2,5
    ORI  x4,x1,2
    XORI x5,x4,5
    SLLI x6,x5,1
    SRLI x7,x2,2
    SRAI x1,x2,2
