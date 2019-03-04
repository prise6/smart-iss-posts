#!/bin/sh


## synchroniser avec les images déjà collectées

rsync -rve ssh my-deblan:~/projets/ISS-HDEV-wallpaper/Collections/ ~/Projets/smart-iss-posts/data/raw/collections/

scp -v ssh my-deblan:~/projets/ISS-HDEV-wallpaper/history.txt ~/Projets/smart-iss-posts/data/raw/history/