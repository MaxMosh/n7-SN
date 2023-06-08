// module minicraps(rst, clk, on, bregin[3..0], adin[7..0] : 
//	abus[31..0], bbus[31..0], rambus[31..0], dbus[31..0], pc[31..0], ir[31..0], flags[3..0], etats[4..0])
  
// avec le prgramme :
//    "00000000" : "0xC2000006",	// set 6, r2
//    "00000001" : "0x83200000",	// load [r2+r0], r3
//	  "00000010" : "0x04310000",	// add r3, r1, r4
//	  "00000011" : "0x94210000",	// store r4, [r2+r1]
//	  "00000110" : "0x12345678",	// donnée à lire
//	  "00000111" : "0x00000000"		// donnée modifiée par le store

// Initialisations
set rst 1
set clk 0
set rst 0
set bregin[3..0] 0000
set adin[7..0] 00000000

// Vérification départ
check etats[4..1] 1000
check pc[31..0] 00000000000000000000000000000000

// Instruction set 6, %r2 : 3 cycles
set on 1
set clk 1
set clk 0
set clk 1
set clk 0
set clk 1
set clk 0

// Vérification des états et du PC
check etats[4..1] 1000
check pc[31..0] 00000000000000000000000000000001

// Vérification du résultat
set on 0
set bregin[3..0] 0010
check bbus[31..0] 00000000000000000000000000000110

// Instruction load [r2+r0], r3 : 4 cycles
set on 1
set clk 1
set clk 0
set clk 1
set clk 0
set clk 1
set clk 0
set clk 1
set clk 0

// Vérification des états et du PC
check etats[4..1] 1000
check pc[31..0] 00000000000000000000000000000010

// Vérification du résultat
set on 0
set bregin[3..0] 0011
check bbus[31..0] 00010010001101000101011001111000

// Instruction add %r3, %r1, %r4 : 3 cycles
set on 1
set clk 1
set clk 0
set clk 1
set clk 0
set clk 1
set clk 0

// Vérification des états et du PC
check etats[4..1] 1000
check pc[31..0] 00000000000000000000000000000011

// Vérification du résultat
set on 0
set bregin[3..0] 0100
check bbus[31..0] 00010010001101000101011001111001

// Instruction store r4,[r2+r1] : 4 cycles
set on 1
set clk 1
set clk 0
set clk 1
set clk 0
set clk 1
set clk 0
set clk 1
set clk 0

// Vérification des états et du PC
check etats[4..1] 1000
check pc[31..0] 00000000000000000000000000000100

// Vérification du résultat
set on 0
set adin[7..0] 00000111
check rambus[31..0] 00010010001101000101011001111001

