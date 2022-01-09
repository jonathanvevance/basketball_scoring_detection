#!/bin/bash
python3 src/server.py &
node src/server/server.js &
serve -s src/build &
wait -n
exit $?
