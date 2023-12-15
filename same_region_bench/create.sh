#!/bin/sh

aws lambda create-function --function-name $1 \
--zip-file fileb://function.zip --handler lambda_test.handler --runtime nodejs8.10 \
--role ROLE_GOES_HERE \
--timeout 300 \
--runtime nodejs8.10 \
--memory-size 2048 \
--region $2
