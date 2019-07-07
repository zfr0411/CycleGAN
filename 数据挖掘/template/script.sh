#!/bin/sh

if [ 0 -eq $# ]
then
	echo "
Before printing, split the pdf file into two parts,
        one sent to a printer with color, the other without color.
Usage: $0 input.pdf offset_of_page_number color_pages ...
"
	exit 1
fi

SPLIT () {
pdftk $1.pdf cat $2 output $1.color.pdf
pdftk $1.pdf cat $3 output $1.gray.pdf
}

fn=$(basename $1 .pdf)
offset=$2
shift 2
page=$*
color=
gray=
last_old=0
for i in $page
do
	first=$(( (${i%%-*}+1+$offset)/2*2-1 ))
	last=$((  (${i##*-}+1+$offset)/2*2   ))
	if [ $last_old -lt $first ]
	then
		color=$color\ $first-$last
	elif [ $last_old -lt $last ]
	then
		color=$color\ $(($last_old+1))-$last
	else
		continue
	fi
	if [ $(($last_old+1)) -lt $(($first-1)) ]
	then
		gray=$gray\ $(($last_old+1))-$(($first-1))
	fi
	last_old=$last
done
gray=$gray\ $(($last_old+1))-end

SPLIT $fn "$color" "$gray"
