; ECS 154A loading.asm 
; Just executes immediate values
MAIN: ; Entry point to program
    LI x1, DATA
    LW x2, 0(x1)
    LW x3, 1(x1)
    LW x4, 2(x1)
    LW x5, 3(x1)
    NOP
    NOP
    NOP
DATA:
    DAT 0x048C
    DAT 0x159D
    DAT 0x26AE
    DAT 0x37BF
