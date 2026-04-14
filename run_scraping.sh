#!/bin/bash
nohup python3 /home/wwdelvalle/scraping.py > /home/wwdelvalle/scraping.log 2>&1 &
echo $!
