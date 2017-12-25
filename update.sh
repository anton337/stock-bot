#!/bin/sh

rm -f data/*
./get-yahoo-quote.sh MSFT
./get-yahoo-quote.sh AAPL
./get-yahoo-quote.sh FB
./get-yahoo-quote.sh HAL
./get-yahoo-quote.sh COP
./get-yahoo-quote.sh OXY
./get-yahoo-quote.sh NFLX
./get-yahoo-quote.sh NVDA
./get-yahoo-quote.sh DVN
./get-yahoo-quote.sh BABA
./get-yahoo-quote.sh IBM
./get-yahoo-quote.sh AAU
./get-yahoo-quote.sh GOOG
./get-yahoo-quote.sh GOOGL
./get-yahoo-quote.sh HPE
./get-yahoo-quote.sh HPQ
./get-yahoo-quote.sh GS
./get-yahoo-quote.sh TWTR
./get-yahoo-quote.sh 0992.HK #Lenovo
./get-yahoo-quote.sh DIS
./get-yahoo-quote.sh ^DJI #Dow Jones
#./get-yahoo-quote.sh CL=F #Oil price
#./get-yahoo-quote.sh GC=F #Gold price

make

