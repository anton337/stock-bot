#!/bin/bash
make;
./stock-bot 4000 0 network.ann;
#for ((number=11;number < 4000;number+=1))
#{
#  sleep .5
#  A="./stock-bot $number network.ann"
#  echo "$A"
#  $A
#}
