#!/bin/sh

rm -f data/*

#Dow Jones stocks
#./get-yahoo-quote.sh MMM
#./get-yahoo-quote.sh AXP
#./get-yahoo-quote.sh BA
./get-yahoo-quote.sh CAT
#./get-yahoo-quote.sh CVX
#./get-yahoo-quote.sh CSCO
#./get-yahoo-quote.sh KO
#./get-yahoo-quote.sh DWDP
#./get-yahoo-quote.sh XOM
#./get-yahoo-quote.sh GE
#./get-yahoo-quote.sh HD
#./get-yahoo-quote.sh INTC
#./get-yahoo-quote.sh JNJ
#./get-yahoo-quote.sh JPM
#./get-yahoo-quote.sh MCD
#./get-yahoo-quote.sh MRK
#./get-yahoo-quote.sh NKE
#./get-yahoo-quote.sh PFE
#./get-yahoo-quote.sh PG
#./get-yahoo-quote.sh TRV
#./get-yahoo-quote.sh UTX
#./get-yahoo-quote.sh UNH
#./get-yahoo-quote.sh VZ
#./get-yahoo-quote.sh V
#./get-yahoo-quote.sh WMT
./get-yahoo-quote.sh MSFT
./get-yahoo-quote.sh AAPL
#./get-yahoo-quote.sh DIS
#./get-yahoo-quote.sh GS
#./get-yahoo-quote.sh IBM
#./get-yahoo-quote.sh ^DJI #Dow Jones

#Some other stocks
./get-yahoo-quote.sh AMZN
#./get-yahoo-quote.sh FB
./get-yahoo-quote.sh HAL
#./get-yahoo-quote.sh COP
#./get-yahoo-quote.sh OXY
#./get-yahoo-quote.sh NFLX
./get-yahoo-quote.sh NVDA
#./get-yahoo-quote.sh DVN
#./get-yahoo-quote.sh BABA
#./get-yahoo-quote.sh AAU
#./get-yahoo-quote.sh GOOG
#./get-yahoo-quote.sh GOOGL
#./get-yahoo-quote.sh HPE
#./get-yahoo-quote.sh HPQ
#./get-yahoo-quote.sh TWTR
#./get-yahoo-quote.sh 0992.HK #Lenovo

#Commodities 
#./get-yahoo-quote.sh CL=F #Oil price
#./get-yahoo-quote.sh GC=F #Gold price

make

