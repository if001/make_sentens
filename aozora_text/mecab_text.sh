#! /bin/sh
##
#テキストファイルをわかち
#


files=`ls | grep .*txt`

for file in $files
do
    nkf -w $file > "utf8_"$file
    mecab -Owakati "utf8_"$file -o "re_"$file
    rm "utf8_"$file
    rm $file
done

exit 0

