#!/bin/bash
make;
./stock-bot 30;
for ((number=31;number < 4000;number+=1))
{
  sleep .5
  A="./stock-bot $number network.ann"
  echo "$A"
  $A
}
