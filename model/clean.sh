#!/bin/bash
find . -type f  -name *.bin | xargs -I {} -P0 rm -rf {}
