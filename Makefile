.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = smartissposts
PYTHON_INTERPRETER = python
SH_INTERPRETER = /bin/sh

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -path ./mysql -prune -o -type f -name "*.py[co]" -exec rm {} +
	find . -path ./mysql -prune -o -type d -name "__pycache__" -exec rm {} + 

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Write requirements.txt
pipreqs: 
	pipreqs --force $(PROJECT_DIR)

## Install Python Dependencies
requirements: test_environment
	pip install -U pip setuptools wheel
	pip install -r requirements.txt
	python -m pip install --upgrade ptvsd

## debug
debug: 
	python -m ptvsd --host 0.0.0.0 --port 3000 --wait -m ${m}

## Write config template
config_template:
	$(PYTHON_INTERPRETER) iss/tools/config_template.py

## start docker
docker_start:
	docker-compose up -d

## stop docker
docker_stop:
	docker-compose stop

## docker exec bash
docker_bash:
	docker exec --user=jovyan -it jupyter-iss /bin/bash


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Sync photos with my refs
sync_collections: iss/data/sync_collections.sh
	/bin/sh iss/data/sync_collections.sh

populate_db:
	$(PYTHON_INTERPRETER) -m iss.exec.bdd

sampling:
	$(PYTHON_INTERPRETER) -m iss.exec.sampling

training:
	$(PYTHON_INTERPRETER) -m iss.exec.training

exec_clustering:
	$(PYTHON_INTERPRETER) -m iss.exec.clustering

posters:
	$(PYTHON_INTERPRETER) -m iss.exec.posters --config-id=1 --generate=1 --poster-id='test'


#################################################################################
# OUTSIDE CONTAINER                                                             #
#################################################################################

maximize_test:
	cp $(PROJECT_DIR)/data/raw/collections/20180211-130001.jpg $(PROJECT_DIR)/data/isr/input/sample/
	docker run -v "$(PROJECT_DIR)/data/isr:/home/isr/data" -v "$(PROJECT_DIR)/../image-super-resolution/weights:/home/isr/weights" -v "$(PROJECT_DIR)/config/config_isr.yml:/home/isr/config.yml" -it isr -d -p -c config.yml

maximize_version_3:
	cp $(PROJECT_DIR)/data/raw/collections/20170815-130001.jpg $(PROJECT_DIR)/data/isr/input/version_3/
	cp $(PROJECT_DIR)/data/raw/collections/20180902-100001.jpg $(PROJECT_DIR)/data/isr/input/version_3/
	docker run -v "$(PROJECT_DIR)/data/isr:/home/isr/data" -v "$(PROJECT_DIR)/../image-super-resolution/weights:/home/isr/weights" -v "$(PROJECT_DIR)/config/config_isr.yml:/home/isr/config.yml" -it isr -d -p -c config.yml


#################################################################################
# FLOYDHUB                                                                      #
#################################################################################

floyd_training:
	floyd run --task training

floyd_training_prod:
	floyd run --task training_prod

floyd_retraining:
	floyd run --task retraining


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
