#!/bin/bash

reset

if [ -d "built" ]; then
	cd built
	rm -R *
else
	mkdir built
	cd built
fi

cmake ..
make
