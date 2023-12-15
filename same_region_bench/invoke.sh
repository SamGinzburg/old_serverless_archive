#!/bin/sh

aws lambda invoke --function-name $1 --log-type Tail \
--region $3 \
--payload "$2" \
outputfile.txt 
