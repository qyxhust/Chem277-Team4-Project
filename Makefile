PYTHONPATH := $(shell pwd)

environment:
	...

clean_data:
	PYTHONPATH=$(PYTHONPATH) python src/clean_data.py

train:
	...
