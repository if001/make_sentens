#!/bin/sh

./get_aozora.sh
if [ $? -ne 0 ]; then
    echo "./get_aozora.sh error"
fi

python3 reshape_text.py
if [ $? -ne 0 ]; then
    echo "./reshape_text.py error"
fi

./linking_text.sh
if [ $? -ne 0 ]; then
    echo "./linking_text.sh error"
fi
