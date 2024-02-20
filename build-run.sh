#!/usr/bin/env bash
#builds all docker images in this mono-repo and runs them using docker-compose

declare -a imageName=()
declare -a imageVersion=()
size=0

for dir in */
do
    currentDir=$(echo "${dir}" | sed 's/\///g')
    imageName[${size}]=${currentDir}
    #echo "${currentDir}"

    #fetch the latest version number of each image
    fullVersion=$(sudo docker images | grep -o -P "${currentDir}.{0,20}" | grep -o -P ' v.{0,5}' | sed 's/ //g' | sed 's/v//g' | sort -V | tail -n 1)
    majorVersion=$(echo "${fullVersion}" | grep -o -P '.{0,10}\.' | sed 's/\.//g')
    minorVersion=$(echo "${fullVersion}" | grep -o -P '\..{0,10}' | sed 's/\.//g')

    #echo "${currentDir}: ${fullVersion}  -  v${majorVersion}.${minorVersion}"

    #increment version no.
    if [ $1 = 'major' ]; then
	((majorVersion++))
    elif [ $1 = 'minor' ]; then
	((minorVersion++))
    fi
    newVersion=$(echo "v${majorVersion}.${minorVersion}")

    imageVersion[${size}]=${newVersion}
    #echo "${newVersion}"

    ((size++))
done

for (( i = 0; i <= $size; i++ ))
do
    #echo "${imageName[$i]}  ${imageVersion[$i]}"
    if find "${imageName[$i]}" -type f -name "Dockerfile" | grep -q .; then
      cd ${imageName[$i]}/ && sudo docker build . -t ${imageName[$i]}:${imageVersion[$i]} -t ${imageName[$i]}:latest && cd ../
    fi
done

sudo docker-compose up -d #runs docker compose headless mode

#sudo docker exec -it off-grid-cctv_backend_1 /bin/bash #this will ssh into the container
