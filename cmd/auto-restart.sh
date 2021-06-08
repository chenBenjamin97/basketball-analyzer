#!/bin/bash
#make sure a process is always running.

process="cmd"
path="/home/cs612/openpose-yolov4/yolov4-deepsort/"
process_exec="./cmd"

if ps ax | grep -v grep | grep $process > /dev/null
then
    exit
else
    (cd $path && $process_exec &)
fi

exit