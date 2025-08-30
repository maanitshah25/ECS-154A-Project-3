#!/usr/bin/env python3
from __future__ import annotations
# SSII25
from enum import IntEnum
from io import TextIOBase
import os
import sys
import argparse
from typing import Dict, List, NamedTuple, Tuple


class CPU15:
    REG_NAME_STR        = 'X'
    REGISTER_COUNT      = 8
    MAX_BRANCH_DELTA    = 2**(9)
    MAX_JUMP_DELTA      = 2**(6)

    _li_table = None

    class Opcode(IntEnum):
        ADD     = 0
        SUB     = 1
        AND     = 2
        OR      = 3
        XOR     = 4
        SLL     = 5
        SRL     = 6
        SRA     = 7
        ADDI    = 8
        LW      = 9
        ANDI    = 10
        ORI     = 11
        XORI    = 12
        SLLI    = 13
        SRLI    = 14
        SRAI    = 15
        BZF     = 16
        BNZF    = 17
        BNF     = 18
        BNNF    = 19
        BCF     = 20
        BNCF    = 21
        RTI     = 22
        SWI     = 23
        SSRP    = 24
        SSRF    = 25
        SF      = 26
        JALR    = 27
        JAL     = 28
        SFI     = 29
        SW      = 30
        LUI     = 31

    class PseudoInstruction(IntEnum):
        NOP     = 0
        LI      = 1
        NOT     = 2
        NEG     = 3
        MV      = 4
        BEQ     = 5
        BNE     = 6
        BLT     = 7
        BGE     = 8
        BLTU    = 9
        BGEU    = 10
        J       = 11
        JR      = 12

    class LoadImmediateSubInstruction(NamedTuple):
        opcode : CPU15.Opcode
        imm : int

    @classmethod
    def get_li_table(cls : CPU15) -> Dict[int,List[CPU15.LoadImmediateSubInstruction]]:
        if cls._li_table is not None:
            return cls._li_table

        def abs_imm(imm : int) -> int:
            return imm if imm >= 0 else 32768 + imm

        def lui_values():
            return [(lui, abs_imm(lui<<4)) for lui in range(-2**6,2**6)]

        def addi_values(base : int = 0):
            return [(i, abs_imm((base + i) & 0x7FFF)) for i in range(-8,8)]

        def xori_values(base : int = 0):
            return [(i, abs_imm(base ^ i)) for i in range(-8,8)]

        def slli_values(base : int, s_low : int = 1, s_high : int = 15):
            return [(i, (base<<i) & 0x7FFF) for i in range(s_low,s_high)]

        def srli_values(base : int, s_low : int = 1, s_high : int = 15):
            return [(i, (base>>i)) for i in range(s_low,s_high)]

        new_li_table = dict()
        for addi_imm,imm in addi_values():
            new_li_table[imm] = [(cls.Opcode.ADDI,addi_imm)]

        for lui_imm,imm in lui_values():
            new_li_table[imm] = [(cls.Opcode.LUI,lui_imm)]
        
        for lui_imm,lui_val in lui_values():
            for xori_imm, imm in xori_values(lui_val):
                if imm not in new_li_table:
                    new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.XORI,xori_imm)]

        for lui_imm,lui_val in lui_values():
            for addi_imm, imm in addi_values(lui_val):
                if imm not in new_li_table:
                    new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.ADDI,addi_imm)]

        for lui_imm,lui_val in lui_values():
            for slli_imm, imm in slli_values(lui_val):
                if imm not in new_li_table:
                    new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.SLLI,slli_imm)]

        for lui_imm,lui_val in lui_values():
            for srli_imm, imm in srli_values(lui_val):
                if imm not in new_li_table:
                    new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.SRLI,srli_imm)]

        for lui_imm,lui_val in lui_values():
            for slli_imm, slli_val in slli_values(lui_val):
                for addi_imm, imm in addi_values(slli_val):
                    if imm not in new_li_table:
                        new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.SLLI,slli_imm),(cls.Opcode.ADDI,addi_imm)]

        for lui_imm,lui_val in lui_values():
            for srli_imm, srli_val in srli_values(lui_val):
                for addi_imm, imm in addi_values(srli_val):
                    if imm not in new_li_table:
                        new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.SRLI,srli_imm),(cls.Opcode.ADDI,addi_imm)]

        for lui_imm,lui_val in lui_values():
            for srli_imm, srli_val in srli_values(lui_val):
                for xori_imm, imm in xori_values(srli_val):
                    if imm not in new_li_table:
                        new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.SRLI,srli_imm),(cls.Opcode.XORI,xori_imm)]

        for lui_imm,lui_val in lui_values():
            for addi_imm, addi_val in addi_values(lui_val):
                for slli_imm, imm in slli_values(addi_val):
                    if imm not in new_li_table:
                        new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.ADDI,addi_imm),(cls.Opcode.SLLI,slli_imm)]
                        
        for lui_imm,lui_val in lui_values():
            for addi_imm1, addi_val1 in addi_values(lui_val):
                for addi_imm2, imm in addi_values(addi_val1):
                    if imm not in new_li_table:
                        new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.ADDI,addi_imm1),(cls.Opcode.ADDI,addi_imm2)]

        for lui_imm,lui_val in lui_values():
            for addi_imm1, addi_val1 in addi_values(lui_val):
                for slli_imm, slli_val in slli_values(addi_val1):
                    for addi_imm2, imm in addi_values(slli_val):
                        if imm not in new_li_table:
                            new_li_table[imm] = [(cls.Opcode.LUI,lui_imm),(cls.Opcode.ADDI,addi_imm1),(cls.Opcode.SLLI,slli_imm),(cls.Opcode.ADDI,addi_imm2)]
        cls._li_table = new_li_table
        """for i, inst_list in sorted(cls._li_table.items()):
            if i and len(inst_list) < len(cls._li_table[i-1]):
                print(f'{i-1} -> {i}:\n    {cls._li_table[i-1]}\n    {inst_list}')"""
        return cls._li_table

    @classmethod
    def expand_li_constant(cls : CPU15, constant : int) -> List[CPU15.LoadImmediateSubInstruction]:
        pos_constant = constant if constant >= 0 else 32768 + constant
        return cls.get_li_table()[pos_constant]


class LineType(IntEnum):
    EMPTY       = 0
    LABEL       = 1
    INSTRUCTION = 2
    PSEUDO      = 3
    DATA        = 4
    INVALID     = 5

class TokenType(IntEnum):
    IDENTIFIER  = 0
    REGISTER    = 1
    INSTRUCTION = 2
    PSEUDO      = 3
    DAT         = 4
    CONSTANT    = 5
    COMMA       = 6
    COLON       = 7
    PLUS        = 8
    MINUS       = 9
    OPEN_PAREN  = 10
    CLOSE_PAREN = 11
    INVALID     = 12

class Token:
    DAT_STR         = 'DAT'

    _register_names                     = set([f'{CPU15.REG_NAME_STR}{i}' for i in range(CPU15.REGISTER_COUNT)])
    _instruction_names                  = set([CPU15.Opcode(i).name for i in range(max(CPU15.Opcode).value+1)])
    _pseudo_instruction_names           = set([CPU15.PseudoInstruction(i).name for i in range(max(CPU15.PseudoInstruction).value+1)])
    _rtype_instruction_names            = set([CPU15.Opcode(i).name for i in range(CPU15.Opcode.ADDI)])
    _itype_instruction_names            = set([CPU15.Opcode(i).name for i in range(CPU15.Opcode.ADDI,CPU15.Opcode.BZF) if i != CPU15.Opcode.LW])
    _btype_instruction_names            = set([CPU15.Opcode(i).name for i in range(CPU15.Opcode.BZF,CPU15.Opcode.RTI)])
    _interrupt_type_instruction_names   = set([CPU15.Opcode(i).name for i in range(CPU15.Opcode.RTI,CPU15.Opcode.SSRP)])
    _swap_type_instruction_names        = set([CPU15.Opcode(i).name for i in range(CPU15.Opcode.SSRP,CPU15.Opcode.JALR)])
    _lwjalr_type_instruction_names      = set([inst.name for inst in [CPU15.Opcode.LW,CPU15.Opcode.JALR]])
    _sfilui_type_instruction_names      = set([inst.name for inst in [CPU15.Opcode.SFI,CPU15.Opcode.LUI]])

    def __init__(self, 
                 line_number : int, 
                 char_position : int, 
                 text : str, 
                 token_type : TokenType):
        self.__line_number = line_number
        self.__char_position = char_position
        self.__raw_text = text
        self.__text = text.upper()
        if token_type == TokenType.IDENTIFIER:
            if  self.text in self._register_names:
                self.__token_type = TokenType.REGISTER
            elif self.text in self._instruction_names:
                self.__token_type = TokenType.INSTRUCTION
            elif self.text in self._pseudo_instruction_names:
                self.__token_type = TokenType.PSEUDO
            elif self.text == self.DAT_STR:
                self.__token_type = TokenType.DAT
            else:
                self.__token_type = token_type
        else:
            self.__token_type = token_type

    def __str__(self):
        return f'@({self.__line_number},{self.__char_position}) {self.__token_type.name} {self.__text}'

    @property
    def line_number(self) -> int:
        return self.__line_number

    @property
    def char_position(self) -> int:
        return self.__char_position

    @property
    def raw_text(self) -> str:
        return self.__raw_text

    @property
    def text(self) -> str:
        return self.__text

    @property
    def token_type(self) -> TokenType:
        return self.__token_type

    @property
    def is_identifier(self) -> bool:
        return self.__token_type == TokenType.IDENTIFIER

    @property
    def is_register(self) -> bool:
        return self.__token_type == TokenType.REGISTER

    @property
    def is_constant(self) -> bool:
        return self.__token_type == TokenType.CONSTANT

    @property
    def is_comma(self) -> bool:
        return self.__token_type == TokenType.COMMA

    @property
    def is_colon(self) -> bool:
        return self.__token_type == TokenType.COLON

    @property
    def is_plus(self) -> bool:
        return self.__token_type == TokenType.PLUS

    @property
    def is_minus(self) -> bool:
        return self.__token_type == TokenType.MINUS

    @property
    def is_open_paren(self) -> bool:
        return self.__token_type == TokenType.OPEN_PAREN

    @property
    def is_close_paren(self) -> bool:
        return self.__token_type == TokenType.CLOSE_PAREN

    @property
    def is_instruction(self) -> bool:
        return self.__token_type == TokenType.INSTRUCTION

    @property
    def is_rtype_instruction(self) -> bool:
        return self.text in self._rtype_instruction_names if self.is_instruction else False

    @property
    def is_itype_instruction(self) -> bool:
        return self.text in self._itype_instruction_names if self.is_instruction else False

    @property
    def is_btype_instruction(self) -> bool:
        return self.text in self._btype_instruction_names if self.is_instruction else False

    @property
    def is_interrupt_type_instruction(self) -> bool:
        return self.text in self._interrupt_type_instruction_names if self.is_instruction else False

    @property
    def is_swap_type_instruction(self) -> bool:
        return self.text in self._swap_type_instruction_names if self.is_instruction else False

    @property
    def is_lwjalr_type_instruction(self) -> bool:
        return self.text in self._lwjalr_type_instruction_names if self.is_instruction else False

    @property
    def is_jal_instruction(self) -> bool:
        return self.text == CPU15.Opcode.JAL.name if self.is_instruction else False

    @property
    def is_sfilui_type_instruction(self) -> bool:
        return self.text in self._sfilui_type_instruction_names if self.is_instruction else False

    @property
    def is_sw_instruction(self) -> bool:
        return self.text == CPU15.Opcode.SW.name if self.is_instruction else False

    @property
    def is_pseudo_instruction(self) -> bool:
        return self.__token_type == TokenType.PSEUDO

    @property
    def is_dat(self) -> bool:
        return self.__token_type == TokenType.DAT

    @property
    def is_keyword(self) -> bool:
        return self.is_instruction or self.is_pseudo_instruction or self.is_dat or self.is_register

    @property
    def constant(self) -> int:
        if self.text.find(CPU15.REG_NAME_STR) < 0:
            return int(self.text)
        return int(self.text,base=16)

class LineTokenizer:
    COMMENT_STR     = ';'
    UNDERSCORE_STR  = '_'
    HEX_CHAR_STR    = 'X'
    COMMA_STR       = ','
    COLON_STR       = ':'
    PLUS_STR        = '+'
    MINUS_STR       = '-'
    OPEN_PAREN_STR  = '('
    CLOSE_PAREN_STR = ')'

    def __init__(self, line_number : int, line : str):
        self.__line_number = line_number
        self.__line = line
        self.__line_length = len(line)
        self.__char_position = 0
        self.__anchor_position = 0

    @property
    def line_number(self) -> int:
        return self.__line_number

    @property
    def line(self) -> str:
        return self.__line
    
    @property
    def line_length(self) -> int:
        return self.__line_length

    @property
    def at_end(self) -> bool:
        return self.__char_position >= self.__line_length

    @property
    def char_position(self) -> int:
        return self.__char_position + 1
    
    @property
    def anchor_position(self) -> int:
        return self.__anchor_position + 1
    
    @property
    def peek_character(self) -> str:
        return self.__line[self.__char_position] if self.__char_position < self.__line_length else ''

    def mark(self):
        self.__anchor_position = self.__char_position

    def consume_char(self):
        self.__char_position = min(self.__char_position + 1, self.__line_length)

    def dispose_char(self):
        self.consume_char()
        self.mark()
    
    def dispose_remaining(self):
        self.__anchor_position = self.__char_position = self.__line_length

    @property
    def current_token(self) -> str:
        return self.__line[self.__anchor_position:self.__char_position] if self.__anchor_position < self.__line_length else ''

    @staticmethod
    def merge_sign_constant(sign : Token, constant : Token) -> Token:
        return Token(line_number=sign.line_number,char_position=sign.char_position,text=sign.raw_text+constant.raw_text,token_type=TokenType.CONSTANT)
    
    @staticmethod
    def merge_sign_constant_list(tokens : List[Token]) -> List[Token]:
        token_index = 0
        while token_index + 1 < len(tokens):
            if (tokens[token_index].is_plus or tokens[token_index].is_minus) and tokens[token_index+1].is_constant:
                tokens[token_index] = LineTokenizer.merge_sign_constant(tokens[token_index],tokens[token_index+1])
                del tokens[token_index+1]
            token_index += 1
        return tokens

    @staticmethod
    def is_first_identifier_char(ch : str) -> bool:
        return ch.isalpha() or ch == LineTokenizer.UNDERSCORE_STR

    @staticmethod
    def is_identifier_char(ch : str) -> bool:
        return ch.isalnum() or ch == LineTokenizer.UNDERSCORE_STR

    def parse_constant(self : LineTokenizer) -> Token:
        self.consume_char()
        token_type = TokenType.CONSTANT
        if self.current_token == '0' and self.peek_character.upper() == self.HEX_CHAR_STR:
            # Hex constant
            self.consume_char()
            while not self.at_end and self.is_identifier_char(self.peek_character):
                if 'F' < self.peek_character.upper():
                    token_type = TokenType.INVALID
                self.consume_char()
            if len(self.current_token) < 3:
                # must be 0x something
                token_type = TokenType.INVALID

        else:
            # Decimal constant
            while not self.at_end and self.is_identifier_char(self.peek_character):
                if not self.peek_character.isdigit():
                    token_type = TokenType.INVALID
                self.consume_char()
        return_token = Token(line_number=self.line_number,
                                char_position=self.anchor_position,
                                text=self.current_token,
                                token_type=token_type)
        self.mark()
        return return_token
    
    def parse_identifier(self : LineTokenizer) -> Token:
        while not self.at_end and self.is_identifier_char(self.peek_character):
            self.consume_char()
        return_token = Token(line_number=self.line_number,
                                char_position=self.anchor_position,
                                text=self.current_token,
                                token_type=TokenType.IDENTIFIER)
        self.mark()
        return return_token

    def parse_remaining_line(self : LineTokenizer) -> List[Token]:
        operators = {self.COMMA_STR:TokenType.COMMA,
                    self.COLON_STR:TokenType.COLON,
                    self.PLUS_STR:TokenType.PLUS,
                    self.MINUS_STR:TokenType.MINUS,
                    self.OPEN_PAREN_STR:TokenType.OPEN_PAREN,
                    self.CLOSE_PAREN_STR:TokenType.CLOSE_PAREN}
        return_tokens : List[Token] = list()
        self.mark()
        while not self.at_end:
            if self.peek_character.isspace():
                self.dispose_char()
                continue
            if self.peek_character == self.COMMENT_STR: 
                # Found a comment, end the parsing
                self.dispose_remaining()
                break
            if self.peek_character in operators:
                self.consume_char()
                return_tokens.append(Token(line_number=self.line_number,
                                           char_position=self.anchor_position,
                                           text=self.current_token,
                                           token_type=operators[self.current_token]))
                self.mark()
                continue
            if self.peek_character.isdigit():
                return_tokens.append(self.parse_constant())
                continue
            if self.is_first_identifier_char(self.peek_character):
                return_tokens.append(self.parse_identifier())
                continue
            self.consume_char()
            return_tokens.append(Token(line_number=self.line_number,
                                        char_position=self.anchor_position,
                                        text=self.current_token,
                                        token_type=TokenType.INVALID))
            self.mark()
        return self.merge_sign_constant_list(return_tokens)

class ProcessedLine:
    def __init__(self, line_number : int, raw_line : str, line_type : LineType, unexpanded_address : int | None = None):
        self.__line_number = line_number
        self.__raw_line = raw_line
        self.__line_type = line_type
        self.__unexpanded_address = unexpanded_address

    @property
    def line_number(self) -> int:
        return self.__line_number

    @property
    def raw_line(self) -> str:
        return self.__raw_line

    @property
    def line_type(self) -> LineType:
        return self.__line_type

    @property
    def unexpanded_address(self) -> int | None:
        return self.__unexpanded_address
    
    @unexpanded_address.setter
    def unexpanded_address(self, address : int | None):
        if self.__unexpanded_address is None:
            self.__unexpanded_address = address

    @property
    def formatted_line(self) -> str:
        return ' '.join(self.raw_line.upper().split())

    @property
    def encoded_line(self) -> int:
        return 0

    @property
    def is_empty(self) -> bool:
        return self.line_type == LineType.EMPTY

    @property
    def is_label(self) -> bool:
        return self.line_type == LineType.LABEL

    @property
    def is_instruction(self) -> bool:
        return self.line_type == LineType.INSTRUCTION

    @property
    def is_pseudo_instruction(self) -> bool:
        return self.line_type == LineType.PSEUDO

    @property
    def is_data(self) -> bool:
        return self.line_type == LineType.DATA

class EmptyLine(ProcessedLine):
    def __init__(self, line_number : int, raw_line : str):
        super().__init__(line_number=line_number,raw_line=raw_line,line_type=LineType.EMPTY)

    @property
    def formatted_line(self) -> str:
        return ''

class LabelLine(ProcessedLine):
    def __init__(self, line_number : int, raw_line : str, label : str):
        super().__init__(line_number=line_number,raw_line=raw_line,line_type=LineType.LABEL)
        self.__label = label

    @property
    def formatted_line(self) -> str:
        return f'{self.label}:'

    @property
    def label(self) -> str:
        return self.__label

class DataLine(ProcessedLine):
    def __init__(self, line_number : int, raw_line : str, data : int):
        super().__init__(line_number=line_number,raw_line=raw_line,line_type=LineType.DATA)
        if data < 0:
            data += 32768
        if data < 0 or data >= 32768:
            raise ValueError(f'Data value out of range on line {line_number}')
        self.__data = data

    @property
    def formatted_line(self) -> str:
        return f'    DAT  0x{self.data:04X}'

    @property
    def encoded_line(self) -> int:
        return self.data

    @property
    def data(self) -> int:
        return self.__data

class InstructionLine(ProcessedLine):
    def __init__(self, 
                 line_number : int, 
                 raw_line : str, 
                 opcode : str, 
                 rd : str | None  = None, 
                 rs1 : str | None = None, 
                 rs2 : str | None  = None, 
                 imm : int | None  = None, 
                 target : str | None  = None, 
                 line_type : LineType | None = None):
        super().__init__(line_number=line_number,
                         raw_line=raw_line,
                         line_type=LineType.INSTRUCTION if line_type is None else line_type)
        self.__opcode = opcode
        self.__rd = rd
        self.__rs1 = rs1
        self.__rs2 = rs2
        self.__imm = imm
        self.__target = target

    @property
    def formatted_line(self) -> str:
        operands = list()
        if self.__rd is not None:
            operands.append(self.__rd)
        if self.__rs2 is not None:
            operands.append(self.__rs2)
        if self.__rs1 is not None:
            operands.append(self.__rs1)
        if self.__imm is not None:
            operands.append(str(self.__imm))
        if self.__target is not None:
            operands.append(self.__target if self.target_is_label else str(self.target_as_constant))
        return f'    {self.__opcode:4} {", ".join(operands)}'

    @property
    def encoded_line(self) -> int:
        encoding = CPU15.Opcode[self.opcode]
        shift_amount = 5
        if self.__rd is not None:
            encoding += int(self.__rd[1])<<shift_amount
            shift_amount += 3
        if self.__rs1 is not None:
            encoding += int(self.__rs1[1])<<shift_amount
            shift_amount += 3
        if self.__rs2 is not None:
            encoding += int(self.__rs2[1])<<shift_amount
            shift_amount += 3
        if self.__imm is not None:
            pos_imm = self.__imm if self.__imm >= 0 else 32768 + self.__imm
            encoding += (pos_imm<<shift_amount) & 0x7FFF
        if self.__target is not None:
            pos_imm = self.target_as_constant if self.target_as_constant >= 0 else 32768 + self.target_as_constant
            encoding += (pos_imm<<shift_amount) & 0x7FFF

        return encoding

    @property
    def opcode(self) -> str:
        return self.__opcode

    @property
    def rd(self) -> str:
        return self.__rd if self.__rd is not None else ''

    @property
    def rs1(self) -> str:
        return self.__rs1 if self.__rs1 is not None else ''

    @property
    def rs2(self) -> str:
        return self.__rs2 if self.__rs2 is not None else ''

    @property
    def imm(self) -> int:
        return self.__imm if self.__imm is not None else 0

    @property
    def target(self) -> str:
        return self.__target if self.__target is not None else ''
    
    @target.setter
    def target(self, target : str):
        self.__target = target

    @property
    def target_is_label(self) -> bool:
        return False if self.__target is None else LineTokenizer.is_first_identifier_char(self.__target[0])

    @property
    def target_as_constant(self) -> int:
        return int(self.__target) if self.__target.find('X') < 0 else int(self.__target,base=16)
    
    def target_delta_is_valid(self, delta_address : int) -> bool: # pragma: no cover
        return False 

class RTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rd : str, rs1 : str, rs2 : str):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rd=rd,rs1=rs1,rs2=rs2)

class ITypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rd : str, rs1 : str, imm : int):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rd=rd,rs1=rs1,imm=imm)

class BTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, target : str):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,target=target)

    def target_delta_is_valid(self, delta_address : int) -> bool:
        return -CPU15.MAX_BRANCH_DELTA <= delta_address < CPU15.MAX_BRANCH_DELTA

class InterruptTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode)

class SwapTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rd : str, rs1 : str):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rd=rd,rs1=rs1)

class LWJALRTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rd : str, rs1 : str, imm : int):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rd=rd,rs1=rs1,imm=imm)

    @property
    def formatted_line(self) -> str:
        return f'    {self.opcode:4} {self.rd}, {self.imm}({self.rs1})'

class JALTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rd : str, target : str):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rd=rd,target=target)

    def target_delta_is_valid(self, delta_address : int) -> bool:
        return -CPU15.MAX_JUMP_DELTA <= delta_address < CPU15.MAX_JUMP_DELTA

class SFILUITypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rd : str, imm : int):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rd=rd,imm=imm)

class SWTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rs2 : str, rs1 : str, imm : int):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rs2=rs2,rs1=rs1,imm=imm)

    @property
    def encoded_line(self) -> int:
        encoding = CPU15.Opcode[self.opcode]
        shift_amount = 5
        encoding += int(self.imm & 0x7)<<shift_amount
        shift_amount += 3
        encoding += int(self.rs1[1])<<shift_amount
        shift_amount += 3
        encoding += int(self.rs2[1])<<shift_amount
        shift_amount += 3
        imm_upper = self.imm>>3
        pos_imm = imm_upper if imm_upper >= 0 else 32768 + imm_upper
        encoding += (pos_imm<<shift_amount) & 0x7FFF

        return encoding
    
class PseudoBTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rs2 : str, rs1 : str, target : str):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rs2=rs2,rs1=rs1,target=target,line_type=LineType.PSEUDO)

    def expand_instructions(self) -> List[ProcessedLine]:
        branch_translations = {CPU15.PseudoInstruction.BEQ.name:CPU15.Opcode.BZF.name,
                                CPU15.PseudoInstruction.BNE.name:CPU15.Opcode.BNZF.name,
                                CPU15.PseudoInstruction.BLT.name:CPU15.Opcode.BNF.name,
                                CPU15.PseudoInstruction.BGE.name:CPU15.Opcode.BNNF.name,
                                CPU15.PseudoInstruction.BLTU.name:CPU15.Opcode.BCF.name,
                                CPU15.PseudoInstruction.BGEU.name:CPU15.Opcode.BNCF.name}
        sub_inst = RTypeLine(line_number=self.line_number,
                             raw_line=self.raw_line,
                             opcode=CPU15.Opcode.SUB.name,
                             rd='X0',
                             rs1=self.rs1,
                             rs2=self.rs2)
        br_inst = BTypeLine(line_number=self.line_number,
                            raw_line='',
                            opcode=branch_translations[self.opcode],
                            target=self.target)
        return [sub_inst,br_inst]

class PseudoLTypeLine(InstructionLine):
    def __init__(self, line_number : int, raw_line : str, opcode : str, rd : str, target : str):
        super().__init__(line_number=line_number,raw_line=raw_line,opcode=opcode,rd=rd,target=target,line_type=LineType.PSEUDO)

    def expand_instructions(self,symbol_table : Dict[str,int] | None = None) -> List[ProcessedLine]:
        if self.target_is_label:
            if symbol_table is None:
                return [self]
            constant_to_use = symbol_table[self.target]
        else:
            constant_to_use = self.target_as_constant
        expanded_list = list()
        for inst_tuple in CPU15.expand_li_constant(constant_to_use):
            opcode,imm = inst_tuple
            if opcode == CPU15.Opcode.LUI:
                expanded_list.append(SFILUITypeLine(line_number=self.line_number,
                                                    raw_line=self.raw_line if len(expanded_list) == 0 else '',
                                                    opcode=opcode.name,
                                                    rd=self.rd,
                                                    imm=imm))
            elif opcode == CPU15.Opcode.ADDI:
                expanded_list.append(ITypeLine(line_number=self.line_number,
                                               raw_line=self.raw_line if len(expanded_list) == 0 else '',
                                               opcode=opcode.name,
                                               rd=self.rd,
                                               rs1=self.rd if len(expanded_list) else 'X0',imm=imm))
            else:
                expanded_list.append(ITypeLine(line_number=self.line_number,
                                               raw_line=self.raw_line if len(expanded_list) == 0 else '',
                                               opcode=opcode.name,
                                               rd=self.rd,
                                               rs1=self.rd,
                                               imm=imm))
        return expanded_list

class CPU15FileProcessor:

    def __init__(self, input_file : TextIOBase):
        self.__line_number = 0
        self.__all_lines = input_file.readlines()

    @property
    def line_number(self) -> int:
        return self.__line_number + 1
    
    @property
    def current_line(self) -> str | None:
        return self.__all_lines[self.__line_number] if self.__line_number < len(self.__all_lines) else None

    def consume_line(self):
        self.__line_number = min(self.__line_number + 1, len(self.__all_lines))

    def _process_token_types(self, tokens : List[Token], expected_types : List[TokenType,Tuple[TokenType,...]]):
        message = None
        if len(tokens) < len(expected_types):
            message = f'Unexpected end of line on line {self.line_number}.'
        else:
            for index, token_type in enumerate(expected_types):
                if (isinstance(token_type,TokenType) and tokens[index].token_type != token_type) or (isinstance(token_type,tuple) and tokens[index].token_type not in token_type):
                    message = f'Unexpected token "{tokens[index].raw_text}" on line {self.line_number}.'
                    break
            if not message and len(tokens) > len(expected_types):
                message = f'Unexpected token "{tokens[len(expected_types)].raw_text}" on line {self.line_number}.'
        if message is not None:
            raise ValueError(message)

    def _process_label(self, tokens : List[Token]) -> LabelLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.IDENTIFIER,TokenType.COLON])
        return LabelLine(line_number=self.line_number,raw_line=self.current_line,label=tokens[0].text)
    
    def _process_dat(self, tokens : List[Token]) -> DataLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.DAT,TokenType.CONSTANT])
        return DataLine(line_number=self.line_number,raw_line=self.current_line,data=tokens[1].constant)

    def _process_rtype_instruction(self, tokens : List[Token]) -> RTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER])
        return RTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rd=tokens[1].text,rs1=tokens[3].text,rs2=tokens[5].text)

    def _process_itype_instruction(self, tokens : List[Token]) -> ITypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.CONSTANT])
        return ITypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rd=tokens[1].text,rs1=tokens[3].text,imm=tokens[5].constant)
    
    def _process_btype_instruction(self, tokens : List[Token]) -> BTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                (TokenType.IDENTIFIER,TokenType.CONSTANT)])
        return BTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,target=tokens[1].text)
    
    def _process_interrupt_instruction(self, tokens : List[Token]) -> InterruptTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION])
        return InterruptTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text)
    
    def _process_swap_instruction(self, tokens : List[Token]) -> SwapTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER])
        return SwapTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rd=tokens[1].text,rs1=tokens[3].text)
    
    def _process_lwjalr_instruction(self, tokens : List[Token]) -> LWJALRTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.CONSTANT,
                                                                TokenType.OPEN_PAREN,
                                                                TokenType.REGISTER,
                                                                TokenType.CLOSE_PAREN])
        return LWJALRTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rd=tokens[1].text,rs1=tokens[5].text,imm=tokens[3].constant)
    
    def _process_jal_instruction(self, tokens : List[Token]) -> JALTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                (TokenType.IDENTIFIER,TokenType.CONSTANT)])
        return JALTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rd=tokens[1].text,target=tokens[3].text)

    def _process_sfilui_instruction(self, tokens : List[Token]) -> SFILUITypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.CONSTANT])
        return SFILUITypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rd=tokens[1].text,imm=tokens[3].constant)
    
    def _process_sw_instruction(self, tokens : List[Token]) -> SWTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.INSTRUCTION,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.CONSTANT,
                                                                TokenType.OPEN_PAREN,
                                                                TokenType.REGISTER,
                                                                TokenType.CLOSE_PAREN])
        return SWTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rs2=tokens[1].text,rs1=tokens[5].text,imm=tokens[3].constant)
    
    def _process_instruction(self, tokens : List[Token]) -> InstructionLine:
        if tokens[0].is_rtype_instruction:
            return self._process_rtype_instruction(tokens=tokens)
        if tokens[0].is_itype_instruction:
            return self._process_itype_instruction(tokens=tokens)
        if tokens[0].is_btype_instruction:
            return self._process_btype_instruction(tokens=tokens)
        if tokens[0].is_interrupt_type_instruction:
            return self._process_interrupt_instruction(tokens=tokens)
        if tokens[0].is_swap_type_instruction:
            return self._process_swap_instruction(tokens=tokens)
        if tokens[0].is_lwjalr_type_instruction:
            return self._process_lwjalr_instruction(tokens=tokens)
        if tokens[0].is_jal_instruction:
            return self._process_jal_instruction(tokens=tokens)
        if tokens[0].is_sfilui_type_instruction:
            return self._process_sfilui_instruction(tokens=tokens)
        if tokens[0].is_sw_instruction:
            return self._process_sw_instruction(tokens=tokens)
        raise ValueError(f'Unknown instruction "{tokens[0].raw_text}" on line {self.line_number}.')  # pragma: no cover

    def _process_nop_pseudo_instruction(self, tokens : List[Token]) -> RTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO])
        return RTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=CPU15.Opcode.ADD.name,rd='X0',rs1='X0',rs2='X0')

    def _process_not_pseudo_instruction(self, tokens : List[Token]) -> ITypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER])
        return ITypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=CPU15.Opcode.XORI.name,rd=tokens[1].text,rs1=tokens[3].text,imm=-1)

    def _process_neg_pseudo_instruction(self, tokens : List[Token]) -> RTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER])
        return RTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=CPU15.Opcode.SUB.name,rd=tokens[1].text,rs1="X0",rs2=tokens[3].text)
    
    def _process_mv_pseudo_instruction(self, tokens : List[Token]) -> ITypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER])
        return ITypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=CPU15.Opcode.ADDI.name,rd=tokens[1].text,rs1=tokens[3].text,imm=0)

    def _process_btype_pseudo_instruction(self, tokens : List[Token]) -> PseudoBTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                (TokenType.IDENTIFIER,TokenType.CONSTANT)])
        return PseudoBTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rs1=tokens[1].text,rs2=tokens[3].text,target=tokens[5].text)

    def _process_li_pseudo_instruction(self, tokens : List[Token]) -> PseudoLTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO,
                                                                TokenType.REGISTER,
                                                                TokenType.COMMA,
                                                                (TokenType.IDENTIFIER,TokenType.CONSTANT)])
        return PseudoLTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=tokens[0].text,rd=tokens[1].text,target=tokens[3].text)

    def _process_j_pseudo_instruction(self, tokens : List[Token]) -> JALTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO,
                                                                (TokenType.IDENTIFIER,TokenType.CONSTANT)])
        return JALTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=CPU15.Opcode.JAL.name,rd='X0',target=tokens[1].text)

    def _process_jr_pseudo_instruction(self, tokens : List[Token]) -> LWJALRTypeLine:
        self._process_token_types(tokens=tokens,expected_types=[TokenType.PSEUDO,
                                                                TokenType.REGISTER])
        return LWJALRTypeLine(line_number=self.line_number,raw_line=self.current_line,opcode=CPU15.Opcode.JALR.name,rd='X0',rs1=tokens[1].text,imm=0)

    def _process_pseudo_instruction(self, tokens : List[Token]) -> InstructionLine:
        if tokens[0].text == CPU15.PseudoInstruction.NOP.name:
            return self._process_nop_pseudo_instruction(tokens=tokens)
        if tokens[0].text == CPU15.PseudoInstruction.NOT.name:
            return self._process_not_pseudo_instruction(tokens=tokens)
        if tokens[0].text == CPU15.PseudoInstruction.NEG.name:
            return self._process_neg_pseudo_instruction(tokens=tokens)
        if tokens[0].text == CPU15.PseudoInstruction.MV.name:
            return self._process_mv_pseudo_instruction(tokens=tokens)
        if tokens[0].text in [CPU15.PseudoInstruction(i).name for i in range(CPU15.PseudoInstruction.BEQ.value,CPU15.PseudoInstruction.BGEU.value+1)]:
            return self._process_btype_pseudo_instruction(tokens=tokens)
        if tokens[0].text == CPU15.PseudoInstruction.LI.name:
            return self._process_li_pseudo_instruction(tokens=tokens)
        if tokens[0].text == CPU15.PseudoInstruction.J.name:
            return self._process_j_pseudo_instruction(tokens=tokens)
        if tokens[0].text == CPU15.PseudoInstruction.JR.name:
            return self._process_jr_pseudo_instruction(tokens=tokens)
        raise ValueError(f'Unknown pseudo instruction "{tokens[0].raw_text}" on line {self.line_number}.')  # pragma: no cover
        
    def _process_line(self) -> ProcessedLine:
        tokenizer = LineTokenizer(line_number=self.line_number,line=self.current_line)
        tokens = tokenizer.parse_remaining_line()
        if len(tokens) == 0:
            return EmptyLine(line_number=self.line_number,raw_line=self.current_line)
        if tokens[0].is_identifier:
            return self._process_label(tokens=tokens)
        if tokens[0].is_dat:
            return self._process_dat(tokens=tokens)
        if tokens[0].is_instruction:
            return self._process_instruction(tokens=tokens)
        if tokens[0].is_pseudo_instruction:
            return self._process_pseudo_instruction(tokens=tokens)

        raise ValueError(f'Unparsable line on {self.line_number} "{self.current_line}"')

    def process_remaining_lines(self : CPU15FileProcessor) -> List[ProcessedLine]:
        processed_lines : List[ProcessedLine] = list()
        while self.current_line is not None:
            processed_lines.append(self._process_line())
            self.consume_line()
        return processed_lines


class CPU15Assembler:
    def __init__(self, input_file : TextIOBase):
        self.__labels : Dict[str,int] = dict()
        self.__instructions_data : List[ProcessedLine] = list()
        self.__resolved_instructions_data : List[ProcessedLine] = list()
        self.__symbol_table : Dict[str,int] = dict()
        self.__reverse_symbol_table : Dict[int,List[str]] = dict()
        self.__unexpanded_symbol_table : Dict[str,int] = dict()
        self.__li_expansions : Dict[int,int] = dict()
        self._load_file(input_file=input_file)

    @property
    def symbol_table(self) -> Dict[str,int]:
        return self.__symbol_table
    
    @property
    def reverse_symbol_table(self) -> Dict[int,List[str]]:
        return self.__reverse_symbol_table

    @property
    def unexpanded_symbol_table(self) -> Dict[str,int]:
        return self.__unexpanded_symbol_table
    
    @property
    def instructions_data(self) -> Tuple[ProcessedLine,...]:
        return tuple(self.__instructions_data)

    @staticmethod
    def unexpanded_address_to_internal_label(address : int) -> str:
        return f'@{address:04X}'

    def _insert_symbol_table_label(self, label : str):
        self.__symbol_table[label] = len(self.__instructions_data)
        labels_at_address = self.__reverse_symbol_table.get(len(self.__instructions_data),list())
        labels_at_address.append(label)
        self.__reverse_symbol_table[len(self.__instructions_data)] = labels_at_address

    def _input_instructions_data_labels(self, input_file : TextIOBase):
        file_processor = CPU15FileProcessor(input_file=input_file)
        unexpanded_address = 0
        for processed_line in file_processor.process_remaining_lines():
            if processed_line.is_empty:
                continue
            if processed_line.is_label:
                if processed_line.label in self.__labels:
                    raise ValueError(f'Repeat label {processed_line.label} on line {processed_line.line_number}')
                else:
                    self.__labels[processed_line.label] = processed_line.line_number
                    self.__unexpanded_symbol_table[processed_line.label] = unexpanded_address
                    self._insert_symbol_table_label(processed_line.label)
                    continue
            unexpanded_address_label = self.unexpanded_address_to_internal_label(unexpanded_address)
            self._insert_symbol_table_label(unexpanded_address_label)
            processed_line.unexpanded_address = unexpanded_address
            if processed_line.is_instruction:
                self.__instructions_data.append(processed_line)
            elif processed_line.is_pseudo_instruction:
                self.__instructions_data.extend(processed_line.expand_instructions())
            elif processed_line.is_data:
                self.__instructions_data.append(processed_line)
            unexpanded_address += 1
    
    def _update_relative_branch_targets(self):
        for processed_line in self.__instructions_data:
            if isinstance(processed_line,(BTypeLine, JALTypeLine)) and not processed_line.target_is_label:
                processed_line.target = self.unexpanded_address_to_internal_label(processed_line.unexpanded_address + processed_line.target_as_constant)

    def _calculate_max_expansion(self, current_address : int, current_target : int, max_expansion : int, processed_line : PseudoLTypeLine) -> int:
        if current_address < current_target:
            temp_symbol_table = {processed_line.target:current_target}
            expansion = processed_line.expand_instructions(temp_symbol_table)
            while max_expansion < len(expansion):
                current_target += len(expansion) - max_expansion
                max_expansion = len(expansion)
                temp_symbol_table = {processed_line.target:current_target}
                expansion = processed_line.expand_instructions(temp_symbol_table)
        else:
            expansion = processed_line.expand_instructions(self.__symbol_table)
            max_expansion = max(max_expansion, len(expansion))
        return max_expansion
    
    def _update_symbol_table_for_delta_expansion(self, delta_expansion : int, current_address : int):
        replacement_addresses = [addr for addr in self.__reverse_symbol_table if addr > current_address]
        for addr in sorted(replacement_addresses,reverse=True):
            new_addr = addr + delta_expansion
            symbols_to_update = self.__reverse_symbol_table[addr]
            del self.__reverse_symbol_table[addr]
            self.__reverse_symbol_table[new_addr] = symbols_to_update
            for symbol in symbols_to_update:
                self.__symbol_table[symbol] = new_addr

    def _update_symbol_table_for_expansion(self):
        current_address = 0
        inst_data_index = 0
        while inst_data_index < len(self.__instructions_data):
            processed_line = self.__instructions_data[inst_data_index]
            max_expansion = 1
            if processed_line.is_pseudo_instruction:
                # check that is PseudoLTypeLine
                if not isinstance(processed_line,PseudoLTypeLine):
                    raise ValueError(f'Unexpected pseudo instruction of type {type(processed_line)}.') # pragma: no cover
                current_target = self.__symbol_table[processed_line.target]
                previous_expansion = max_expansion = self.__li_expansions.get(processed_line.unexpanded_address,1)
                max_expansion = self._calculate_max_expansion(current_address=current_address,current_target=current_target,max_expansion=max_expansion,processed_line=processed_line)
                self.__li_expansions[processed_line.unexpanded_address] = max_expansion
                if max_expansion > previous_expansion:
                    # need to update symbols
                    self._update_symbol_table_for_delta_expansion(delta_expansion=max_expansion - previous_expansion,current_address=current_address)
                    if len(self.__li_expansions) > 1:
                        # start over again
                        inst_data_index = 0
                        current_address = 0
                        continue
            inst_data_index += 1
            current_address += max_expansion


    def _resolve_pseudo_instruction(self, processed_line : PseudoLTypeLine):
        # check that is PseudoLTypeLine
        if not isinstance(processed_line,PseudoLTypeLine):
            raise ValueError(f'Unexpected pseudo instruction of type {type(processed_line)}.') # pragma: no cover
        # li
        expansion = processed_line.expand_instructions(self.__symbol_table)
        while len(expansion) < self.__li_expansions[processed_line.unexpanded_address]:
            # Insert NOP
            new_nop = RTypeLine(line_number=processed_line.line_number,
                                       raw_line=processed_line.raw_line,
                                       opcode=CPU15.Opcode.ADD.name,
                                       rd='X0',
                                       rs1='X0',
                                       rs2='X0')
            new_nop.unexpanded_address = processed_line.unexpanded_address
            expansion.append(new_nop)
        self.__resolved_instructions_data.extend(expansion)

    def _resolve_target_instruction(self, processed_line : InstructionLine):
        target_address = self.__symbol_table[processed_line.target]
        delta_address = target_address - len(self.__resolved_instructions_data)
        if not processed_line.target_delta_is_valid(delta_address=delta_address):
            raise ValueError(f'{processed_line.opcode} at line {processed_line.line_number} too far away!')
        if isinstance(processed_line,BTypeLine):
            new_processed_line = BTypeLine(line_number=processed_line.line_number,
                                           raw_line=processed_line.raw_line,
                                           opcode=processed_line.opcode,
                                           target=str(delta_address))
        else:
            new_processed_line = JALTypeLine(line_number=processed_line.line_number,
                                             raw_line=processed_line.raw_line,
                                             opcode=processed_line.opcode,
                                             rd=processed_line.rd,
                                             target=str(delta_address))
        new_processed_line.unexpanded_address = processed_line.unexpanded_address
        self.__resolved_instructions_data.append(new_processed_line)

    def _resolve_instructions_data(self):
        for processed_line in self.__instructions_data:
            if processed_line.is_data:
                self.__resolved_instructions_data.append(processed_line)
            elif isinstance(processed_line,InstructionLine) and not processed_line.target_is_label:
                self.__resolved_instructions_data.append(processed_line)
            elif processed_line.is_pseudo_instruction:
                self._resolve_pseudo_instruction(processed_line=processed_line)
            else:
                self._resolve_target_instruction(processed_line=processed_line)

    def _load_file(self, input_file : TextIOBase):
        self._input_instructions_data_labels(input_file=input_file)
        self._update_relative_branch_targets()
        self._update_symbol_table_for_expansion()
        self._resolve_instructions_data()

        #self._initialize_symbol_table()
        #self._update_symbol_table_for_expansions()

    def pretty_print(self):
        for address, proc_line in enumerate(self.__resolved_instructions_data):
            for symbol in self.__reverse_symbol_table.get(address,list()):
                if '@' != symbol[0]:
                    print(f'{symbol}:')
            buffer = ' '*(16 - len(proc_line.formatted_line.strip()))
            print(f'{address:04X} {proc_line.encoded_line:04X}: {proc_line.formatted_line}{buffer}; {proc_line.raw_line.strip()}')

    def output_file(self,output_file : TextIOBase):
        out_lines = ['v3.0 hex words addressed']
        address = 0
        current_line = list()
        for proc_line in self.__resolved_instructions_data:
            current_line.append(f'{proc_line.encoded_line:04x}')
            if len(current_line) == 16:
                out_lines.append(f'{address:04x}: {" ".join(current_line)}')
                current_line = list()
                address += 16
        if len(current_line):
            out_lines.append(f'{address:04x}: {" ".join(current_line)}')
        output_file.writelines('\n'.join(out_lines))

    @staticmethod
    def main(*argv):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('-v','--verbose',action='store_true')
        arg_parser.add_argument('-d','--disable',action='store_true')
        arg_parser.add_argument('-o','--output',default=None)
        arg_parser.add_argument('src_file')
        arguments = arg_parser.parse_args(list(argv))

        with open(arguments.src_file,'r') as in_file:
            assembler = CPU15Assembler(input_file=in_file)
            if arguments.verbose:
                assembler.pretty_print()

            if not arguments.disable:
                filename = f'{os.path.splitext(arguments.src_file)[0]}.dat' if arguments.output is None else arguments.output
                print(f'Saving to {filename}')
                with open(filename,'w') as out_file:
                    assembler.output_file(output_file=out_file)
            
            return 0
        
if __name__ == '__main__': # pragma: no coverage
    sys.exit(CPU15Assembler.main(*sys.argv[1:]))